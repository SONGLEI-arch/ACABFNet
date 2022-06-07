import numpy as np

'''
输入anno为单张单通道索引图，类型为array
输出彩色RGB图像
'''
def create_visual_anno1(anno):
    """"""
    assert np.max(anno) <= 2
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 255, 255],
        # 0: [255, 255, 255],
        # 1: [0, 128, 0],
        # 2: [128, 128, 128],
        # 3: [0, 255, 0],
        # 4: [0, 0, 255],
        # 5: [128, 0, 0],
        # 6: [255, 0, 0]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

def create_visual_anno2(anno):
    """"""
    assert np.max(anno) <= 6
    label2color_dict = {
        0: [255, 255, 255],
        1: [0, 128, 0],
        2: [128, 128, 128],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [128, 0, 0],
        6: [255, 0, 0]
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno