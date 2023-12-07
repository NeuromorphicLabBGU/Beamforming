import numpy as np
import iterations as it
import neurons_types as sctn_t


def array_of(f, n):
    return [f() for _ in range(n)]


type0 = array_of(sctn_t.create_type0, n=2)
type1 = array_of(sctn_t.create_type1, n=2)
type1a = array_of(sctn_t.create_type1a, n=2)
type2 = array_of(sctn_t.create_type2, n=1)
type3 = array_of(sctn_t.create_type3, n=62)
type4 = array_of(sctn_t.create_type4, n=16)
type5 = array_of(sctn_t.create_type5, n=8)
type6 = array_of(sctn_t.create_type6, n=4)
type7 = array_of(sctn_t.create_type7, n=2)
type8 = array_of(sctn_t.create_type8, n=1)
type9 = array_of(sctn_t.create_type9, n=1)

# need to get encoded but whatever..
# pdm1 = np.sin(np.linspace(0, 10 * 2 * np.pi, 1000))
# pdm2 = np.sin(np.linspace(0, 10 * 2 * np.pi, 1000))
pdm1 = np.ones(160)
pdm2 = np.zeros(160)

shift_reg_inp1 = np.zeros(9)
shift_reg_inp1[0::2] = 1
shift_reg_inp2 = np.zeros(9)
shift_reg_inp2[0::2] = 1
pulse_dly_buff1 = np.zeros(16)
pulse_dly_buff2 = np.zeros(16)
pulse_dly_buff_pntr = 0
for i in range(len(pdm1)):
    pdm1_byte = pdm1[i]
    pdm2_byte = pdm2[i]
    [o0_1, o0_2] = it.iteration0(pdm1_byte, pdm2_byte, type0)
    it.shift_register(shift_reg_inp1, pdm1_byte, o0_1)
    it.shift_register(shift_reg_inp2, pdm2_byte, o0_2)
    [o1_1, o1_2] = it.iteration1(shift_reg_inp1, shift_reg_inp2, type1)
    [o1a_1, o1a_2] = it.iteration1a(o1_1, o1_2, type1a)
    print(o1a_1)
    [o2_1] = it.iteration2(o1_1, type2)

    if (i + 1) % len(pulse_dly_buff1) != 0:
        continue
    pulse_dly_buff1[pulse_dly_buff_pntr] = o1a_1
    pulse_dly_buff2[pulse_dly_buff_pntr] = o1a_2
    out3 = it.iteration3(pulse_dly_buff_pntr, pulse_dly_buff1, pulse_dly_buff2, o1a_1, o1a_2, type3)
    out4 = it.iteration4(out3, type4)
    out5 = it.iteration5(out3, type5)
    out6 = it.iteration6(out3, type6)
    out7 = it.iteration7(out3, type7)
    out8 = it.iteration8(out3, type8)
    out9 = it.iteration9(type9, o2_1, out4, out5, out6, out7, out8)
    # print(out9, end=',')
    pulse_dly_buff_pntr = (pulse_dly_buff_pntr+1) % 16
    i += 1
