import torch


def convert_s_to_p(s_coord, wtrans, ptrans, prot):
    if type(s_coord) == list:
        p_coord_lst = []
        for s_coord_i in s_coord:
            p_coord_lst.append(convert_s_to_p(s_coord_i[0], wtrans, ptrans, prot)[None])
        return p_coord_lst

    # 从s坐标系到w坐标系
    w_coord = s_coord - wtrans

    # 从w坐标系到o坐标系
    o_coord = w_coord[:, [0, 2, 1]]
    o_coord[:, 2] *= -1

    # 从o坐标系到p坐标系
    p_coord = (o_coord - ptrans) @ prot

    return p_coord