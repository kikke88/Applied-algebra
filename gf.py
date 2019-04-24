import numpy

def gen_pow_matrix(primpoly):
    q = primpoly.bit_length() - 1
    max_num = 2 ** q
    remainder = primpoly - max_num
    matrix = numpy.empty((2 ** q - 1, 2), dtype = int)
    lambda_x = 2;
    for i in range(1, 2 ** q):
        matrix[i - 1][1] = lambda_x
        matrix[lambda_x - 1][0] = i
        lambda_x <<= 1
        if lambda_x >= max_num:
            lambda_x -= max_num
            lambda_x ^= remainder
    return matrix

def add(X, Y):
    return X ^ Y

def sum(X, axis = 0):
    rows, columns = X.shape
    if axis:
        size = rows
    else:
        size = columns
    result = numpy.zeros(size, dtype = int)
    for i in range(rows):
        for j in range(columns):
            if axis:
                result[i] ^= X[i][j]
            else:
                result[j] ^= X[i][j]        
    return result

def prod(X, Y, pm):
    dim = X.ndim
    is_one_dim = False
    if dim == 1:
        X = numpy.array([X])
        Y = numpy.array([Y])
        is_one_dim = True
    res = numpy.empty(X.shape, dtype = int)
    q = pm.shape[0]
    rows, columns = X.shape
    for i in range(rows):
        for j in range(columns):
            if X[i][j] == 0 or Y[i][j] == 0:
                res[i][j] = 0
            else:
                x_pow = pm[X[i][j] - 1][0]
                y_pow = pm[Y[i][j] - 1][0]
                x_pow = (x_pow + y_pow) % q
                res[i][j] = pm[x_pow - 1][1]
    if is_one_dim:
        res = res[0]
    return res

def prod_zero_dim(X, Y, pm):
    q = pm.shape[0]
    if X == 0 or Y == 0:
        res = 0
    else:
        x_pow = pm[X - 1][0]
        y_pox = pm[Y - 1][0]
        x_pow = (x_pow + y_pox) % q
        res = pm[x_pow - 1][1]
    return res

def divide(X, Y, pm):
    dim = X.ndim
    is_one_dim = False
    if dim == 1:
        X = numpy.array([X])
        Y = numpy.array([Y])
        is_one_dim = True
    res = numpy.empty(X.shape, dtype = int)
    q = pm.shape[0]
    rows, columns = X.shape
    for i in range(rows):
        for j in range(columns):
            if Y[i][j] == 0:
                return numpy.nan
            elif X[i][j] == 0:
                res[i][j] = 0
            else:
                x_pow = pm[X[i][j] - 1][0]
                y_pox = pm[Y[i][j] - 1][0]
                x_pow = (x_pow - y_pox) % q
                res[i][j] = pm[x_pow - 1][1]
    if is_one_dim:
        res = res[0]
    return res
 
def divide_zero_dim(X, Y, pm):
    q = pm.shape[0]
    if Y == 0:
        return numpy.nan
    elif X == 0:
        res = 0
    else:
        x_pow = pm[X - 1][0]
        y_pox = pm[Y - 1][0]
        x_pow = (x_pow - y_pox) % q
        res = pm[x_pow - 1][1]
    return res

def linsolve(A, b, pm):
    tmp_A = A.copy()
    tmp_b = b.copy()
    size = tmp_A.shape[0]
    q = pm.shape[0]
    tmp_row = numpy.zeros(size, dtype = int)
    minus_row = numpy.zeros(size, dtype = int)
    answ = numpy.zeros(size, dtype = int)
    for k in range(size):
        if tmp_A[k][k] == 0:
            not_zero_line = k
            for i in range(k + 1, size):
                if tmp_A[i][k] != 0:
                    not_zero_line = i
                    break
            if not_zero_line == k:
                return numpy.nan
            tmp = tmp_A[k].copy()
            tmp_A[k] = tmp_A[not_zero_line]
            tmp_A[not_zero_line] = tmp
            tmp = tmp_b[k].copy()
            tmp_b[k] = tmp_b[not_zero_line]
            tmp_b[not_zero_line] = tmp
            del tmp
        first_elem = tmp_A[k][k]
        tmp_row += first_elem
        tmp_A[k] = divide(tmp_A[k], tmp_row, pm)
        tmp_b[k] = divide_zero_dim(tmp_b[k], first_elem, pm)
        tmp_row -= first_elem
        for j in range(k + 1, size):
            tmp_first = tmp_A[j][k].copy()
            minus_row += tmp_first
            tmp_A[j] = add(tmp_A[j], prod(tmp_A[k], minus_row, pm))
            tmp_b[j] = add(tmp_b[j], prod_zero_dim(tmp_b[k], tmp_first, pm))
            minus_row -= tmp_first
            del tmp_first
    for k in range(size - 1, -1, -1):
        summ = 0
        prod_row = prod(tmp_A[k][k + 1:], answ[k + 1:], pm)
        summ = sum(numpy.array([prod_row]), 1)[0]
        answ[k] = tmp_b[k] ^ summ
    return answ 


def minpoly(x, pm):
    roots_list = []
    for elem in x.flat:
        if elem in roots_list:
            continue
        roots_list.append(elem)
        elem_in_pow = prod_zero_dim(elem, elem, pm)
        while elem_in_pow not in roots_list:
            roots_list.append(elem_in_pow)
            elem_in_pow = prod_zero_dim(elem_in_pow, elem_in_pow, pm)
    roots_list.sort()
    all_x = numpy.array(roots_list)
    poly = polyprod(numpy.array([1, all_x[0]]), numpy.array([1, all_x[1]]), pm)
    for i in range(2, all_x.shape[0]):
        poly = polyprod(poly, numpy.array([1, all_x[i]]), pm)
    return poly, all_x

def polyval(p, x, pm):
    res = numpy.empty(x.shape[0], dtype = int)
    for j, elem in enumerate(x):
        cur = 1
        cur_sum = 0
        for i in range(p.shape[0] - 1, -1, -1):
            cur_sum ^= prod_zero_dim(p[i], cur, pm)
            cur = prod_zero_dim(cur, elem, pm)
        res[j] = cur_sum
    return res

def polyprod(p1, p2, pm):
    size_1 = p1.shape[0]
    size_2 = p2.shape[0]
    res_size = size_1 + size_2 - 1
    res = numpy.zeros(res_size, dtype = int)
    for i in range(size_1):
        for j in range(size_2):
            res[i + j] ^= prod_zero_dim(p1[i], p2[j], pm)
    return res

def polydiv(p1, p2, pm):
    size_1 = p1.shape[0]    
    size_2 = p2.shape[0]
    res_size = size_1 - size_2 + 1
    res = numpy.empty(res_size, dtype = int)
    up = p1.copy()
    down = numpy.zeros(size_1, dtype = int)
    for i in range(res_size):
        res[i] = divide_zero_dim(up[i], p2[0], pm)
        for j in range(size_2):
            down[i + j] = prod_zero_dim(res[i], p2[j], pm)
        up = add(up, down)
        down = numpy.zeros(size_1, dtype = int)
    j = 0
    for x in up:
        if x == 0:
            j += 1
        else:
            break
    up = up[0 + j:]
    if numpy.array_equal(up, numpy.array([])):
        up = numpy.array([0])
    return res, up

def euclid(p1, p2, pm, max_deg = 0):
    r_list = [p1, p2, 0]
    y_list = [0, numpy.array([1]), 0]
    quo, rem = polydiv(r_list[0], r_list[1], pm)
    if rem.shape[0] == 0:
        if p1.shape[0] > p2.shape[0]:
            return p2, [0], [1]
        else:
            return p1, [1], [0]
    r_list[2] = rem
    y_list[2] = quo
    if rem.shape[0] <= max_deg + 1:
        x = polyprod(y_list[2], p2, pm)
        tmp_x = numpy.hstack((numpy.array([0 for i in range(x.shape[0] - rem.shape[0])]), rem))
        x = add(tmp_x, x)
        x = polydiv(x, p1, pm)[0]
        return rem, x, y_list[2]
    while rem.shape[0] > max_deg + 1:
        r_list.pop(0)
        y_list.pop(0)
        quo, rem = polydiv(r_list[0], r_list[1], pm)
        r_list.append(rem)
        tmp1 = polyprod(y_list[1], quo, pm)
        if tmp1.shape[0] < y_list[0].shape[0]:
            tmp2 = numpy.hstack((numpy.array([0 for i in range(y_list[0].shape[0] - tmp1.shape[0])]) , tmp1))
            y_list.append(add(y_list[0], tmp2))
        elif tmp1.shape[0] > y_list[0].shape[0]:
            tmp2 = numpy.hstack((numpy.array([0 for i in range(tmp1.shape[0] - y_list[0].shape[0])]) , y_list[0]))
            y_list.append(add(tmp1, tmp2))
        else:
            y_list.append(add(tmp1, y_list[0]))
    x = polyprod(y_list[2], p2, pm)
    tmp_x = numpy.hstack((numpy.array([0 for i in range(x.shape[0] - rem.shape[0])]), rem))
    x = add(tmp_x, x)
    x = polydiv(x, p1, pm)[0]
    return rem, x, y_list[2]    
