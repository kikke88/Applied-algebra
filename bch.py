import numpy
import gf

class BCH:
	def __init__(self, n, t):
		self.n = n
		self.t = t
		all_polynoms = numpy.loadtxt('primpoly.txt', dtype = int, delimiter = ',')
		for poly in all_polynoms:
			poly = int(poly)
			cur_poly_pow = poly.bit_length() - 1
			if 2 ** cur_poly_pow - 1 == n:
				self.polynom = poly
				break
		self.pm = gf.gen_pow_matrix(self.polynom)
		roots = list()
		alpha = 2
		cur = 2
		for i in range(2 * self.t):
			roots.append(cur)
			cur = gf.prod_zero_dim(cur, alpha, self.pm)
		self.R_for_pgz = numpy.array(roots)
		roots.reverse()
		self.R = numpy.array(roots)
		roots.reverse()
		self.g = gf.minpoly(numpy.array(roots), self.pm)[0]
		for i in range(2 * self.t, self.n):
			roots.append(cur)
			cur = gf.prod_zero_dim(cur, alpha, self.pm)
		self.all_roots = numpy.array(roots)
		self.m = self.g.shape[0] - 1
		self.k = self.n - self.m	
	
	def encode(self, U):
		num_of_message = U.shape[0]
		answ = numpy.zeros((num_of_message, self.n), dtype = int)
		for i in range(num_of_message):
			for j in range(self.k):
				answ[i][j] = U[i][j]
			rem = gf.polydiv(answ[i], self.g, self.pm)[1]
			rem = numpy.hstack((numpy.zeros((self.m - rem.shape[0]), dtype = int), rem))
			answ[i][self.k:] = rem
		return answ

	def correction(self, Lambda, Q):
		lambda_val_in_alpha = gf.polyval(Lambda, self.all_roots, self.pm)
		num_of_lambda_root = 0
		root_position = []
		for j in range(self.n):
			if lambda_val_in_alpha[j] == 0:
				num_of_lambda_root += 1
				root_position.append(j)
		e_row = numpy.zeros(self.n, dtype = int)
		for k in root_position:
			e_row[k] = 1
		return Q ^ e_row, num_of_lambda_root

	def decode(self, W, method = 'euclid'):
		num_of_message = W.shape[0]
		answ = numpy.empty(W.shape, dtype = int)
		double_t = 2 * self.t
		for i in range(num_of_message):
			if method == 'euclid':
				syndrome_arr = gf.polyval(W[i], self.R, self.pm)
				if not any(syndrome_arr):
					answ[i] = W[i]
					continue
				syndrome_arr = numpy.hstack((syndrome_arr, numpy.array([1], dtype = int)))
				z_in_pow_poly = numpy.zeros(double_t + 2, dtype = int)
				z_in_pow_poly[0] = 1
				zero_counter = 0
				for x in syndrome_arr:
					if x == 0:
						zero_counter += 1
					else:
						break
				Lambda = gf.euclid(z_in_pow_poly, syndrome_arr[zero_counter:], self.pm, self.t)[2]
				Lambda_pow = Lambda.shape[0] - 1
				answ[i], num_of_root = self.correction(Lambda, W[i])
				if Lambda_pow != num_of_root:
					answ[i]= numpy.nan
					#answ[i] = 2 Программы со сбором статистики, построением графиков, ожидают,
					#что при отказах будет возвращаться строка двоек из-за проблем со сравнением numpy.nan находящегося в целочисленном массиве
					continue
			else:
				syndrome_arr = gf.polyval(W[i], self.R_for_pgz, self.pm)
				if not any(syndrome_arr):
					answ[i] = W[i]
					continue
				arr = numpy.empty((self.t, self.t), dtype = int)
				for k in range(self.t):
					arr[k] = syndrome_arr[k:k + self.t]
				b_arr = numpy.array(syndrome_arr[self.t:])
				j = 0
				while j != self.t:
					Lambda = gf.linsolve(arr[:self.t - j,:self.t - j], b_arr, self.pm)
					if not numpy.array_equal(Lambda, Lambda):
						j += 1
						b_arr = syndrome_arr[self.t - j:2 * (self.t - j)]
					else:
							break
				if j == self.t:
					answ[i] = numpy.nan
					#answ[i] = 2
					continue
				Lambda = numpy.hstack((Lambda, numpy.array([1])))
				answ[i] = self.correction(Lambda, W[i])[0]
				if any(gf.polyval(answ[i], self.R_for_pgz, self.pm)):
					answ[i] = numpy.nan
					#answ[i] = 2
		return answ

	def dist(self):
		min_dist = self.n + 1
		cur_block = numpy.empty((1, self.k), dtype = int)
		for i in range(1, 2 ** self.k):
			cur_num = numpy.array([int(x) for x in bin(i)[2:]], dtype = int)
			cur_block[0] = numpy.hstack((numpy.zeros(self.k - cur_num.shape[0], dtype = int), cur_num))
			cur_code = self.encode(cur_block)
			summ = sum(cur_code[0])
			min_dist = min(min_dist, summ)
		return min_dist
