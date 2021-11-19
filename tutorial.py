import functools
import itertools
import math
from collections import defaultdict

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class CFG:
    n_strings = 3
    letters = [
        "🎅",  # father christmas
        "🤶",  # mother christmas
        "🦌",  # reindeer
        "🧝",  # elf
        #    "🎄",  # christmas tree
        #    "🎁",  # gift
        #    "🎀",  # ribbon
    ]
    wildcard = "🌟"  # star
    n = len(letters)
    depot = itertools.repeat("0", n)  # a starting dummy node
    inf = 999  # number to represent "infinite distance", could try sys.maxsize for hard constraint

    # 検索のソリューション制限
    n_solutions = None
    # 時間制限
    n_minutes = 1


def make_nodes(cfg):
    """
    順列を生成
    node -> 順列

    Describes
    ----------
    🎅🤶で始まる順列がすべてのスケジュールに含まれている
    必要があるという条件を適用するために、
    スケジュールごとにそのようなノードのセットを作成します
    """
    perms = list(itertools.permutations(cfg.letters, cfg.n))
    all_ = perms[: math.factorial(cfg.n - 2)]  # permutations beginning with 🎅🤶
    some = perms[math.factorial(cfg.n - 2) :]  # everything else
    return [cfg.depot] + (all_ * cfg.n_strings) + some


def create_data_model(cfg):
    data = {}
    data["locations"] = make_nodes(cfg)
    data["num_vehicles"] = cfg.n_strings
    data["depot"] = 0
    return data


data = create_data_model(CFG)
# node, 車の数、スタート位置を指示
manager = pywrapcp.RoutingIndexManager(
    len(data["locations"]), data["num_vehicles"], data["depot"]
)
routing = pywrapcp.RoutingModel(manager)


def distance(p, q):
    """
    ノード間の距離を測定する

    Parameters
    ----------
    p, q: ノード（有向）
    depot: ダミーノード
    n: 文字数
    inf: 最大距離

    Descrives
    ---------
    ノード間の最大距離を無限大（または実際には大きな数）に
    設定することにより、検索スペースを削減します。
    この変更は、ノード🎅🤶🦌🧝🎄🎁🎀からノード🧝🎄🎁🎀🎅🤶🦌に
    移動することを決して考慮しないことを意味します。
    """
    if p == CFG.depot or q == CFG.depot:
        return 0
    for num in range(CFG.n):  # never choose maximum distance nodes (the N+1 case)
        if p[num:] == q[: CFG.n - num]:
            return CFG.n
    return CFG.inf  # max distance N becomes distance infinity


def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    from_perm = data["locations"][from_node]
    to_node = manager.IndexToNode(to_index)
    to_perm = data["locations"][to_node]
    return distance(from_perm, to_perm)


# 距離callbackを作成してソルバーに渡す
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

dimension_name = "Length"
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    10 ** 16,  # vehicle maximum travel distance
    True,  # start cumul to zero
    dimension_name,
)
length_dimention = routing.GetDimensionOrDie(dimension_name)
length_dimention.SetGlobalSpanCostCoefficient(
    100
)  # total cost += 100 * (max_length - min_length)

# Set constraint that each vehicle must have all permutations beginning with 🎅🤶
# TODO: 制約を渡してるっぽいけどわからん
# TODO: 初期解を設定して計算早くなる？
# TODO: 開始位置と終了位置も設定できるらしいので考える
n_all = math.factorial(CFG.n - 2)
for vehicle in range(CFG.n_strings):
    for node in range(n_all):
        routing.SetAllowedVehiclesForIndex(
            [vehicle],
            manager.NodeToIndex(1 + vehicle * n_all + node),
        )

# エッジを削除
def remove_forbidden_edges(forbidden_fn):
    n_removed = 0
    n_nodes = len(data["locations"])

    for from_index, to_index in itertools.product(range(n_nodes), range(n_nodes)):
        from_node = manager.IndexToNode(from_index)
        from_perm = data["locations"][from_node]
        to_node = manager.IndexToNode(to_index)
        to_perm = data["locations"][to_node]

        if routing.NextVar(from_index).Contains(to_index) and forbidden_fn(
            from_perm, to_perm
        ):
            n_removed += 1
            routing.NextVar(from_index).RemoveValue(to_index)
    print("Removed", n_removed, "edges.")


# TODO: ここ色々試してみる
# remove_forbidden_edges(
#     lambda p, q: distance(p, q) >= CFG.inf
# )

# 検索の戦略を変更するrouting option
# TODO: どんな設定になっているか分からん
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

# 終了条件を設定
if CFG.n_solutions is not None:
    search_parameters.solution_limit = CFG.n_solutions
if CFG.n_minutes is not None:
    search_parameters.time_limit.seconds = CFG.n_minutes * 60

search_parameters.log_search = True
solution = routing.SolveWithParameters(search_parameters)


# 提出物作成
def get_routes(data, manager, routing, solution):
    routes = defaultdict(list)
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            idx_node = manager.IndexToNode(index)
            if idx_node != data["depot"]:
                routes[vehicle_id].append(idx_node)
            index = solution.Value(routing.NextVar(index))
    return routes


def route_to_schedule(route, nodes):
    def overlap(a, b):
        return max(i for i in range(len(b) + 1) if a.endswith(b[:i]))

    def squeeze(ws):
        return functools.reduce(lambda a, b: a + b[overlap(a, b) :], ws)

    return squeeze(["".join(nodes[i]) for i in route])


def get_schedules(routes):
    words = [
        route_to_schedule(routes[vehicle_id], data["locations"])
        for vehicle_id in range(data["num_vehicles"])
    ]
    return words


if solution:
    routes = get_routes(data, manager, routing, solution)
    words = get_schedules(routes)

    submission = pd.Series(words, name="schedule")
    submission.to_csv("submission.csv", index=False)

    print(submission)
    print(submission.apply(len).rename("Length"))
