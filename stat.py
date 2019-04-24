#построение графиков скорости, подсчёт времени работы программы, метод стат испытаний
import numpy
import numpy.random as rand
import time
import matplotlib.pyplot as plt
import gf
import bch

def speed():
	all_polynoms = numpy.loadtxt('primpoly.txt', dtype = int, delimiter = ',')
	for n in [7, 15, 31, 63]:
		for poly in all_polynoms:
			poly = int(poly)
			cur_poly_pow = poly.bit_length() - 1
			if 2 ** cur_poly_pow - 1 == n:
				polynom = poly
				break
		pm = gf.gen_pow_matrix(polynom)
		roots = list()
		alpha = 2
		cur = 2
		r = numpy.zeros((n - 1) >> 1)
		all_t_for_n = [t for t in range(1, ((n - 1) >> 1) + 1)]
		for t in all_t_for_n:
			roots = []
			for j in range(2 * t):
				roots.append(cur)
				cur = gf.prod_zero_dim(cur, alpha, pm)
			cur = 2
			g = gf.minpoly(numpy.array(roots), pm)[0]
			k = n - (g.shape[0] - 1)
			r[t - 1] = k / n
		plt.plot(all_t_for_n, r)
		plt.title('code length, n = ' + str(n))
		plt.xlabel('number of correctioned errors, t')
		plt.ylabel('code speed, r')
		plt.grid(color = 'k', linestyle = '-', linewidth = 0.5)
		plt.show()
	return

#speed()

def codetime(r):
	n_t_arr = numpy.array([[7, 15, 31, 63, 63, 127, 127],
							[1, 3, 5, 4, 11, 12, 21]], dtype = int)
	time_Euclid = numpy.empty(n_t_arr.shape[1])
	time_pgz = numpy.empty(n_t_arr.shape[1])
	N = 150
	for i in range(n_t_arr.shape[1]):
		cur_bch = bch.BCH(n_t_arr[0][i], n_t_arr[1][i])
		block = rand.randint(0, 2, (N, cur_bch.k))
		code = cur_bch.encode(block)
		if r == 1:
			err_index = rand.randint(0, cur_bch.n, N)
			err_arr = numpy.zeros((N, cur_bch.n), dtype = int)
			for k, j in enumerate(err_index):
				err_arr[k][j] = 1
		else:
			tmp = numpy.arange(cur_bch.n)
			err_index = numpy.empty((N, n_t_arr[1][i]), dtype = int)
			for k in range(N):
				rand.shuffle(tmp)
				err_index[k] = tmp[0:n_t_arr[1][i]]
			err_arr = numpy.zeros((N, cur_bch.n), dtype = int)
			for k in range(N):
				for j in err_index[k]:
					err_arr[k][j] = 1
		bad_code = code ^ err_arr
		time_Euclid[i] = time.clock()
		res = cur_bch.decode(bad_code, 'euclid')
		time_Euclid[i] = (time.clock() - time_Euclid[i])
		if not numpy.array_equal(res, code):
			print("AVOST")
		time_pgz[i] = time.clock()
		res = cur_bch.decode(bad_code, 'pgz')
		time_pgz[i] = (time.clock() - time_pgz[i])
		if not numpy.array_equal(res, code):
			print("AVOST")
			return
	print(time_Euclid)
	print(time_pgz)
	return
# Если аргумент единица, делает одну ошибку, иначе - t
#codetime(1)

def code_stat(n, t):
	stat = numpy.empty((3, n))
	N = 100
	cur_bch = bch.BCH(n, t)
	block = rand.randint(0, 2, (N, cur_bch.k))
	code = cur_bch.encode(block)
	tmp = numpy.arange(cur_bch.n)
	for real_err in range(1, n + 1):
		err_arr = numpy.zeros((N, cur_bch.n), dtype = int)
		for k in range(N):
			rand.shuffle(tmp)
			for i in range(real_err):
				err_arr[k][tmp[i]] = 1
		bad_code = code ^ err_arr
		res = cur_bch.decode(bad_code, 'euclid')
		nice, err, rej = 0, 0, 0

		for i in range(N):
			if numpy.array_equal(code[i], res[i]):
				nice += 1
			elif res[i][0] == 2:
				rej += 1
			else:
				err += 1
		stat[0][real_err - 1] = nice / N * 100
		stat[1][real_err - 1] = rej / N * 100
		stat[2][real_err - 1] = err / N * 100
	print(stat)
	all_err = [int(r) for r in range(1, n + 1)]
	# l1 = plt.plot(all_err, stat[0], color = 'b', linewidth = 1.2, label = 'decode')
	# l2 = plt.plot(all_err, stat[1], color = 'c', linewidth = 1.2, label = 'rejection')
	# l3 = plt.plot(all_err, stat[2], color = 'r', linewidth = 1.2, label = 'error')
	plt.bar(all_err, stat[0] + stat[1] + stat[2], linewidth = 1.2, label = 'error')
	plt.bar(all_err, stat[0] + stat[1], linewidth = 1.2, label = 'rejection')
	plt.bar(all_err, stat[0], linewidth = 1.2, label = 'decode')
	plt.title('(n = ' + str(n) + ', t = ' + str(t) + ')')
	plt.legend()
	ax = plt.axes()
	ax.set_xticks(numpy.arange(1, n + 1, 3))
	plt.ylabel('share of all code words, %')
	plt.xlabel('number of errors, t')
	plt.axvline(x = t, color = 'g', linestyle = '-', linewidth = 1)
	plt.tight_layout()
	plt.show()
	return

#code_stat(7, 1)
#code_stat(15, 3)
#code_stat(31, 5)
#code_stat(63, 11)
