def write_hessian_helper_V1V4_optim():
    axes = ["X", "Y", "Z"]
    print('i3 = 3*i')
    # print('i3 = 3*i')
    # print('i3 = 3*i')
    print('j3 = 3*j')
    # print('j3 = 3*j')
    # print('j3 = 3*j')
    for a in ["i", "j"]:
        for b in ["i", "j"]:
            if a == b:
                sign = 1
            else:
                sign = -1
            i_n = a
            j_n = b
            for a_n, a_str in zip([0, 1, 2], axes):
                for b_n, b_str in zip([0, 1, 2], axes):
                    if sign == -1:
                        print('hessian[{0}3 + {1}][{2}3 + {3}] += {6} * const * bcoord["{4}"] * bcoord["{5}"]'.format(
                            i_n, a_n, j_n, b_n, a_str, b_str, sign))
                    else:
                        print('hessian[{0}3 + {1}][{2}3 + {3}] += const * bcoord["{4}"] * bcoord["{5}"]'.format(
                            i_n, a_n, j_n, b_n, a_str, b_str, sign))


if __name__ == "__main__":
    write_hessian_helper_V1V4_optim()