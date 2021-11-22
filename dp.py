"""
https://www.kaggle.com/koya346/wildcard-postprocessing-using-dynamic-programming/edit
"""

import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def calc_wildcard_dp(schedule):
    # すべての順列
    perms = list(map(lambda p: "".join(p), itertools.permutations("1234567")))
    # 順列をIDに変換
    perm2id = {p: i for i, p in enumerate(perms)}

    # 要素を1文字ずつint型に変換し、リスト化
    perms_arr = np.array([list(map(int, p)) for p in perms])

    #### create cost matrix ####

    # perms_arr-1: perms_arrの要素をindex化
    # np.eye: 単位行列
    # 対角行列をperms_arr-1のindexに従って入れ替え、
    # perms_arrのone hot化を行う
    # transposeの必要性は分からない
    perms_onehot = np.eye(7)[perms_arr - 1, :].transpose(0, 2, 1)

    # one hot化が成功したかチェック
    # np.allclose: すべての値が近いか判定
    assert np.allclose(
        perms_onehot[:, 0, :].astype(np.int64), (perms_arr == 1).astype(np.int64)
    )

    M = (
        F.conv2d(
            F.pad(torch.Tensor(perms_onehot[:, None, :, :]), (7, 7)),
            torch.Tensor(perms_onehot[:, None, :, :]),
            padding="valid",
        )
        .squeeze()
        .numpy()
    )
    must_match_left2right = np.array(
        [-1, -1, -1, -1, -1, -1, -1, 7, 6, 5, 4, 3, 2, 1, 0]
    )
    must_match_left2right_wild = np.array(
        [-1, -1, -1, -1, -1, -1, -1, 6, 5, 4, 3, 2, 1, 0, 0]
    )

    cost_ifmatch = np.array([7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7])

    costMat = (
        np.where(M == must_match_left2right, cost_ifmatch, np.inf)
        .min(axis=-1)
        .astype(np.int8)
    )
    costMatWild = np.minimum(
        costMat,
        np.where(M == must_match_left2right_wild, cost_ifmatch, np.inf).min(axis=-1),
    ).astype(np.int8)
    symbols = "🎅🤶🦌🧝🎄🎁🎀"
    words = [s.translate(str.maketrans(symbols, "1234567")) for s in schedule]

    # Optim
    nodes_list = []
    table_list = []
    for i in range(3):
        word = words[i]
        nodes = []
        for i in range(len(word) - 6):
            p = word[i : i + 7]
            if p in perm2id:
                nodes.append(perm2id[p])
        table = np.zeros((len(nodes), 10), np.int64)
        table[0, :] = 7
        for i in range(1, len(nodes)):
            e = costMat[nodes[i - 1], nodes[i]]
            ew = costMatWild[nodes[i - 1], nodes[i]]
            table[i, 0] = table[i - 1, 0] + e
            table[i, 1] = min(table[i - 1, 1] + e, table[i - 1, 0] + ew)
            table[i, 2] = (
                min(table[i - 1, 2], table[i - 1, 1]) + e
            )  # TODO: better transition
            table[i, 3] = min(table[i - 1, 3], table[i - 1, 2]) + e
            table[i, 4] = min(table[i - 1, 4], table[i - 1, 3]) + e
            table[i, 5] = min(table[i - 1, 5], table[i - 1, 4]) + e
            table[i, 6] = min(table[i - 1, 6], table[i - 1, 5]) + e
            table[i, 7] = min(table[i - 1, 7], table[i - 1, 6]) + e
            table[i, 8] = min(table[i - 1, 8], table[i - 1, 7]) + e
            table[i, 9] = min(table[i - 1, 9] + e, table[i - 1, 8] + ew)
        nodes_list.append(nodes)
        table_list.append(table)

    # backtrack
    new_words = []
    wilds = []
    for nodes, table in zip(nodes_list, table_list):
        ns = [perms[nodes[-1]]]
        track = np.argmin(table[-1])
        wild = []
        for i in range(len(nodes) - 2, -1, -1):
            e = costMat[nodes[i], nodes[i + 1]]
            ew = costMatWild[nodes[i], nodes[i + 1]]
            if track == 0:
                ns.append(perms[nodes[i]][:e])
            elif track == 1:
                if table[i, 1] + e < table[i, 0] + ew:
                    ns.append(perms[nodes[i]][:e])
                else:
                    left = np.array(list(map(int, perms[nodes[i]][ew:])))
                    right = np.array(list(map(int, perms[nodes[i + 1]][:-ew])))
                    mis = np.where(left != right)[0][0]
                    wild.append(table[i, track - 1] - 7 + ew + mis)
                    ns.append(perms[nodes[i]][:ew])
                    track = track - 1
            elif 2 <= track <= 8:
                if table[i, track] >= table[i, track - 1]:
                    track = track - 1
                ns.append(perms[nodes[i]][:e])
            elif track == 9:
                if table[i, 9] + e < table[i, 8] + ew:
                    ns.append(perms[nodes[i]][:e])
                else:
                    ns.append(perms[nodes[i]][:ew])
                    left = np.array(list(map(int, perms[nodes[i]][ew:])))
                    right = np.array(list(map(int, perms[nodes[i + 1]][:-ew])))
                    mis = np.where(left != right)[0][0]
                    wild.append(table[i, track - 1] - 7 + ew + mis)
                    track = track - 1
            else:
                assert False
        assert track == 0
        wilds.append(wild)
        nsw = list("".join(ns[::-1]))
        for w in wild:
            nsw[w] = "*"
        new_words.append("".join(nsw))

    # score
    score = max(map(len, new_words))
    return score, new_words
