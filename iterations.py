import numpy as np


def shift_register(reg: np.array, new_sample: int, dc: int):
    reg[1:] = reg[:-1]
    reg[0] = new_sample
    reg[8] = dc
    return reg


def iteration0(pdm1_byte, pdm2_byte, type0):
    o0 = type0[0].ctn_cycle(np.array([pdm1_byte]), True)
    o1 = type0[1].ctn_cycle(np.array([pdm2_byte]), True)
    return [o0, o1]


def iteration1(reg1, reg2, type1):
    o0 = type1[0].ctn_cycle(reg1, True)
    o1 = type1[1].ctn_cycle(reg2, True)
    return [o0, o1]


def iteration1a(inp1, inp2, type1a):
    o0 = type1a[0].ctn_cycle(inp1, True)
    o1 = type1a[1].ctn_cycle(inp2, True)
    return [o0, o1]


def iteration2(inp1, type2):
    o0 = type2[0].ctn_cycle(inp1, True)
    return [o0]


def iteration3(pulse_dly_buff_pntr, buffer1, buffer2, o1a_1, o1a_2, type3):
    out = [0] * len(type3)
    for dly in range(16):
        f = np.array([
            buffer1[(16 + pulse_dly_buff_pntr - dly) % 16],
            o1a_2,
        ])
        out[2*dly] = type3[2*dly].ctn_cycle(f, True)

        f = np.array([
            o1a_2,
            buffer1[(16 + pulse_dly_buff_pntr - dly) % 16],
        ])
        out[2*dly+1] = type3[2*dly+1].ctn_cycle(f, True)

    for dly in range(1, 16):
        f = np.array([
            buffer2[(16 + pulse_dly_buff_pntr - dly) % 16],
            o1a_1,
        ])
        out[2*dly+30] = type3[2*dly+30].ctn_cycle(f, True)

        f = np.array([
            o1a_1,
            buffer2[(16 + pulse_dly_buff_pntr - dly) % 16],
        ])
        out[2*dly+31] = type3[2*dly+31].ctn_cycle(f, True)
    return out


def iteration4(out3, type4):
    inp = np.zeros((len(type4), 4))
    for dly in range(16):
        inp[dly//2, (2*dly) % 4] = out3[2*dly]
        inp[dly//2, 1 + (2*dly) % 4] = out3[2*dly + 1]
        inp[dly//2 + 8, (2*dly) % 4] = out3[2*dly + 30]
        inp[dly//2 + 8, 1 + (2*dly) % 4] = out3[2*dly + 31]

    inp[8, 0] = out3[0]
    inp[8, 1] = out3[1]
    out = [0] * len(type4)
    for i, n in enumerate(type4):
        out[i] = n.ctn_cycle(inp[i], True)
    return out


def iteration5(out3, type5):
    inp = np.zeros((len(type5), 8))
    for dly in range(16):
        inp[dly//4, (2*dly) % 8] = out3[2*dly]
        inp[dly//4, 1 + (2*dly) % 8] = out3[2*dly + 1]
        inp[dly//4 + 4, (2*dly) % 8] = out3[2*dly + 30]
        inp[dly//4 + 4, 1 + (2*dly) % 8] = out3[2*dly + 31]

    inp[4, 0] = out3[0]
    inp[4, 1] = out3[1]
    out = [0] * len(type5)
    for i, n in enumerate(type5):
        out[i] = n.ctn_cycle(inp[i], True)
    return out


def iteration6(out3, type6):
    inp = np.zeros((len(type6), 16))
    for dly in range(16):
        inp[dly//8, (2*dly) % 16] = out3[2*dly]
        inp[dly//8, 1 + (2*dly) % 16] = out3[2*dly + 1]
        inp[dly//8 + 2, (2*dly) % 16] = out3[2*dly + 30]
        inp[dly//8 + 2, 1 + (2*dly) % 16] = out3[2*dly + 31]

    inp[2, 0] = out3[0]
    inp[2, 1] = out3[1]
    out = [0] * len(type6)
    for i, n in enumerate(type6):
        out[i] = n.ctn_cycle(inp[i], True)
    return out


def iteration7(out3, type7):
    inp = np.zeros((len(type7), 32))
    for dly in range(16):
        inp[0, 2*dly] = out3[2*dly]
        inp[0, 1 + 2*dly] = out3[2*dly + 1]
        inp[1, 2*dly] = out3[2*dly + 30]
        inp[1, 1 + 2*dly] = out3[2*dly + 31]

    inp[1, 0] = out3[0]
    inp[1, 1] = out3[1]
    out = [0] * len(type7)
    for i, n in enumerate(type7):
        out[i] = n.ctn_cycle(inp[i], True)
    return out


def iteration8(out3, type8):
    inp = np.zeros((len(type8), 64))
    for dly in range(16):
        inp[0, 2*dly] = out3[2*dly]
        inp[0, 1 + 2*dly] = out3[2*dly + 1]
        inp[0, 2*dly + 32] = out3[2*dly + 30]
        inp[0, 33 + 2*dly] = out3[2*dly + 31]

    inp[0, 0] = out3[0]
    inp[0, 1] = out3[1]
    out = [0] * len(type8)
    for i, n in enumerate(type8):
        out[i] = n.ctn_cycle(inp[i], True)
    return out


def iteration9(type9, out2: list, out4: list, out5: list, out6: list, out7: list, out8: list):
    sel_1xxxx = (1-out8[0])
    sel_00xxx = (1-sel_1xxxx) & out7[1]
    sel_01xxx = (1-sel_1xxxx) & (1-out7[1])
    sel_10xxx = sel_1xxxx & (1-out7[0])
    sel_11xxx = sel_1xxxx & out7[0]
    sel_000xx = sel_00xxx & out6[3]
    sel_001xx = sel_00xxx & (1-out6[3])
    sel_010xx = sel_01xxx & out6[2]
    sel_011xx = sel_01xxx & (1-out6[2])
    sel_100xx = sel_10xxx & (1-out6[0])
    sel_101xx = sel_10xxx & out6[0]
    sel_110xx = sel_11xxx & (1-out6[1])
    sel_111xx = sel_11xxx & out6[1]
    sel_0000x = sel_000xx & out5[7]
    sel_0001x = sel_000xx & (1-out5[7])
    sel_0010x = sel_001xx & out5[6]
    sel_0011x = sel_001xx & (1-out5[6])
    sel_0100x = sel_010xx & out5[5]
    sel_0101x = sel_010xx & (1-out5[5])
    sel_0110x = sel_011xx & out5[4]
    sel_0111x = sel_011xx & (1-out5[4])
    sel_1000x = sel_100xx & (1-out5[0])
    sel_1001x = sel_100xx & out5[0]
    sel_1010x = sel_101xx & (1-out5[1])
    sel_1011x = sel_101xx & out5[1]
    sel_1100x = sel_110xx & (1-out5[2])
    sel_1101x = sel_110xx & out5[2]
    sel_1110x = sel_111xx & (1-out5[3])
    sel_1111x = sel_111xx & out5[3]
    sel_00001 = sel_0000x & (1-out4[15])
    sel_00011 = sel_0001x & (1-out4[14])
    sel_00101 = sel_0010x & (1-out4[13])
    sel_00111 = sel_0011x & (1-out4[12])
    sel_01001 = sel_0100x & (1-out4[11])
    sel_01011 = sel_0101x & (1-out4[10])
    sel_01101 = sel_0110x & (1-out4[9])
    sel_01111 = sel_0111x & (1-out4[8])
    sel_10001 = sel_1000x & out4[0]
    sel_10011 = sel_1001x & out4[1]
    sel_10101 = sel_1010x & out4[2]
    sel_10111 = sel_1011x & out4[3]
    sel_11001 = sel_1100x & out4[4]
    sel_11011 = sel_1101x & out4[5]
    sel_11101 = sel_1110x & out4[6]
    sel_11111 = sel_1111x & out4[7]

    inp = np.zeros(31)
    inp[0] = sel_1xxxx
    inp[1] = sel_01xxx
    inp[2] = sel_11xxx
    inp[3] = sel_001xx
    inp[4] = sel_011xx
    inp[5] = sel_101xx
    inp[6] = sel_111xx
    inp[7] = sel_0001x
    inp[8] = sel_0011x
    inp[9] = sel_0101x
    inp[10] = sel_0111x
    inp[11] = sel_1001x
    inp[12] = sel_1011x
    inp[13] = sel_1101x
    inp[14] = sel_1111x
    inp[15] = sel_00001
    inp[16] = sel_00011
    inp[17] = sel_00101
    inp[18] = sel_00111
    inp[19] = sel_01001
    inp[20] = sel_01011
    inp[21] = sel_01101
    inp[22] = sel_01111
    inp[23] = sel_10001
    inp[24] = sel_10011
    inp[25] = sel_10101
    inp[26] = sel_10111
    inp[27] = sel_11001
    inp[28] = sel_11011
    inp[29] = sel_11101
    inp[30] = sel_11111

    type9[0].ctn_cycle(inp, out2)


