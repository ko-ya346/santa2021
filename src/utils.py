from itertools import permutations
from typing import List


def get_perm(strings: List[str], required_permutations: List[str]) -> List[List[str]]:
    """
    3つのstringsに追加しなければいけないスケジュールのリストを返す

    Parameters
    ----------
    strings:
        3行の文字列
    required_permutations:
        先頭2文字固定されているスケジュール
    """
    ans = [[] for _ in range(3)]
    for i in range(3):
        for p in required_permutations:
            if p not in strings[i]:
                # スケジュールが含まれていない場合はlistに追加
                ans[i].append(p)
    return ans


# def add_suffix(strings, to_add):
#    for i, (s, permutations) in enumerate(zip(strings, to_add)):
#        for l in range(5, 0, -1):
#            added = False
#            for p in permutations:
#                if p[-l:] == s[:l]:
#                    # 先頭とかぶる部分
#                    strings[i] = p[:l] + strings[i]
#                    to_add[i].remove(p)
#                    added = True
#                    break
#
