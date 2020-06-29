import numpy as np
import matplotlib.pyplot as plt
import random


def draw_graphics(array_of_x_and_y, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["red", "blue", "green", "orange", "violet"]
    for i in range(0, 5):
        ax.plot(array_of_x_and_y[i].p, np.array(array_of_x_and_y[i].r), color=colors[i], linestyle='-', linewidth=1,
                label=array_of_x_and_y[i].name)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, np.max(array_of_x_and_y[0].p))
    ax.set_ylim(0, 1)

    ax.grid(which='major',
            color='black',
            linewidth=1)

    ax.minorticks_on()

    ax.grid(which='minor',
            color='gray',
            linestyle=':')

    fig.tight_layout()
    ax.legend()
    plt.show()


def factorial(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f


def binomial(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


class CyclicCode:
    def __init__(self, n, k, d, name):
        self.name = name
        self.n = n
        self.k = k
        self.d = d
        # list of probability values [min, min + step, ... , max - step, max]
        self.p = list()
        # list of summary values
        self.p_of_correct_intake = list()
        # list of r effective values
        self.r = list()
        self.g = 0
        self.G = list()
        self.H = list()
        self.A = list()

    def generate_p(self, min, max, step):
        self.p = np.arange(min, max + step, step)

    def generate_p_of_correct_intake(self):
        for p in self.p:
            buf = 0
            for i in range(0, self.k):
                c = binomial(self.n, i)
                buf += c * pow(p, i) * pow((1 - p), (self.n - i))
            self.p_of_correct_intake.append(buf)

    def generate_r(self):
        for i in range(0, len(self.p_of_correct_intake)):
            buf = (self.k * self.p_of_correct_intake[i]) / self.n
            self.r.append(buf)

    def generate_G_and_A(self):
        G = np.zeros([self.k, self.n])
        for i in range(0, self.k):
            G[i][i] = 1
        for i in range(0, self.k):
            q, r = np.polydiv(G[i], self.g)
            for k in range(0, len(r)):
                buff = r[k] % 2
                r[k] = buff
            new_result = np.polyadd(G[i], r)
            G[i] = new_result
            self.A.append(r)
        self.G = G

    def generate_H(self):
        r = self.n - self.k
        H = np.zeros([r, self.n])
        A = np.array(self.get_A())
        A_T = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
        self.show_matrix(A_T, "A_T")
        for i in range(0, r):
            for k in range(0, self.n):
                if k < self.k:
                    H[i][k] = A_T[i][k]
            H[i][self.k + i] = 1

        self.H = H

    def generate_g(self, r):
        # gen g(x)
        self.g = np.zeros(r + 1)
        # reverse init because polynomials in python work weird
        for i in range(0, r + 1):
            # 63 36 11
            if i == 27:
                self.g[i] = 1
            if i == 26:
                self.g[i] = 1
            if i == 23:
                self.g[i] = 1
            if i == 19:
                self.g[i] = 1
            if i == 12:
                self.g[i] = 1
            if i == 10:
                self.g[i] = 1
            if i == 9:
                self.g[i] = 1
            if i == 8:
                self.g[i] = 1
            if i == 6:
                self.g[i] = 1
            if i == 5:
                self.g[i] = 1
            if i == 0:
                self.g[i] = 1
            # end 63 36 11
            # 7 4 3
            # if i == 3:
            #     self.g[i] = 1
            # if i == 2:
            #     self.g[i] = 1
            # if i == 0:
            #     self.g[i] = 1
            # end 7 4 3

    def normalize_A(self):
        r = self.n - self.k
        for i in range(0, len(self.A)):
            while len(self.A[i]) < r:
                item = np.insert(self.A[i], 0, 0)
                self.A[i] = item

    def get_A(self):
        return self.A

    def get_H(self):
        return self.H

    def show_matrix(self, matrix, name):
        print(f"\nMATRIX {name}:\n")
        for i in range(0, len(matrix)):
            for k in range(0, len(matrix[i])):
                print(int(matrix[i][k]), end=" ")
            print('\n')

    def generate_error_vector(self, pr_error):
        error_vector = list()
        for count in range(0, self.n):
            rand_num = random.random()
            if rand_num <= pr_error:
                error_vector.append(1)
            else:
                error_vector.append(0)
        return error_vector

    def encode(self, encode_message):
        # generate r ( r = deg (g(x)) )
        r = list()
        for i in range(0, len(self.g)):
            if i == 0:
                r.append(1)
            else:
                r.append(0)
        q, c = np.polydiv(np.polymul(encode_message, r), self.g)
        # GF(2), may be do not use
        for k in range(0, len(c)):
            buff = c[k] % 2
            c[k] = buff
        a = np.polyadd(np.polymul(encode_message, r), c)
        return a

    def decode(self, decode_message):
        #error_vector = self.generate_error_vector(0)
        error_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        print(f"\nError vector:\n\n{error_vector}\n")
        b = list()
        for i in range(0, len(decode_message)):
            buf = int(decode_message[i]) ^ error_vector[i]
            b.append(buf)
        print(f"XOR:\n\n{b}\n")
        t = int((self.d - 1) / 2)
        for i in range(0, t):
            q, c = np.polydiv(b, self.g)
            # normalize c for GF(2)
            c = self.normalize_c(c)
            if np.all(c == 0):
                print("Decode message:\n")
                self.show_decode_message(b)
                break
            else:
                w = self.get_weight(c)
                if w <= t:
                    print("Find error in CRC. Try to decode message\n")
                    b = np.polyadd(b, c)
                    for k in range(0, len(b)):
                        buff = b[k] % 2
                        b[k] = int(buff)
                    print("Fixed:\n")
                    self.show_encode_message(b)
                    print("\n")
                    print("WARNING. CRC is not correct, may be need to receive message again.\n")
                else:
                    try:
                        print("Find error in information part. Try to decode message\n")
                        n = self.gen_number_of_bit(c)
                        bit = 0 if b[n] == 1 else 1
                        b[n] = bit
                        print("Fixed:\n")
                        self.show_encode_message(b)
                        print("\n")
                    except TypeError:
                        print("Can not fix the error. Decode failed.\n")
                        break
        q, c = np.polydiv(b, self.g)
        c = self.normalize_c(c)
        if np.all(c != 0):
            print("Find fatal error. Decode failed.\n")

    def gen_number_of_bit(self, row):
        H = self.get_H()
        for i in range(0, self.n):
            col = H[:, i]
            if np.array_equal(col, row):
                return i

    def get_weight(self, vector):
        weight = 0
        for i in range(0, len(vector)):
            if vector[i] != 0:
                weight += vector[i]
        return weight

    def normalize_c(self, c):
        for i in range(0, len(c)):
            tmp = c[i] % 2
            c[i] = int(tmp)
        r = self.n - self.k
        while len(c) < r:
            c = np.insert(c, 0, 0)
        return c

    def show_decode_message(self, message_for_show):
        tmp = list()
        for count in range(0, len(message_for_show) - (self.n - self.k)):
            if message_for_show[count] > 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
            #tmp.append(message_for_show[count])
        print(tmp)

    @staticmethod
    def show_encode_message(message_for_show):
        tmp = list()
        for count in range(0, len(message_for_show)):
            if message_for_show[count] > 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        print(tmp)



# looking for the best cyclic code
# codes = list()
# cyclic_code_12 = CyclicCode(63, 45, 7, "63, 45, 7")
# codes.append(cyclic_code_12)
# cyclic_code_13 = CyclicCode(63, 39, 9, "63, 39, 9")
# codes.append(cyclic_code_13)
# cyclic_code_14 = CyclicCode(63, 36, 11, "63, 36, 11")
# codes.append(cyclic_code_14)
# cyclic_code_15 = CyclicCode(63, 30, 13, "63, 30, 13")
# codes.append(cyclic_code_15)
# cyclic_code_16 = CyclicCode(63, 24, 15, "63, 24, 15")
# codes.append(cyclic_code_16)
# for item in codes:
#     item.generate_p(0.01, 1, 0.01)
#     item.generate_p_of_correct_intake()
#     item.generate_r()
# #draw_d_systems(codes, "Эффективная скорость прохождения информации для циклических кодов при различных n k d", "p", "r_эф")
# end looking for the best cyclic code

# work with 63 36 11

# x^27 + x^22 + x^21 + x^19 + x^18 + x^17 + x^15 + x^8 + x^4  + x + 1
cyclic_code_14 = CyclicCode(63, 36, 11, "63, 36, 11")
cyclic_code_14.generate_g(27)
cyclic_code_14.generate_G_and_A()
cyclic_code_14.normalize_A()
cyclic_code_14.generate_H()
cyclic_code_14.show_matrix(cyclic_code_14.A, "A")
cyclic_code_14.show_matrix(cyclic_code_14.G, "G")
cyclic_code_14.show_matrix(cyclic_code_14.H, "H")
message = list()
for i in range(0, cyclic_code_14.k):
    num = random.random()
    if num <= 0.5:
        message.append(1)
    else:
        message.append(0)
message[0] = 1
print(f"\nMessage:\n\n{message}\n")
message_encode = cyclic_code_14.encode(message)
print("\nEncode message:\n")
cyclic_code_14.show_encode_message(message_encode)
cyclic_code_14.decode(message_encode)
# end work with 63 36 11

# test (work with 7 4 3)

# hamming = CyclicCode(7, 4, 3, "7, 4, 3")
# hamming.generate_g(3)
# hamming.generate_G_and_A()
# hamming.normalize_A()
# hamming.generate_H()
# hamming.show_matrix(hamming.A, "A")
# hamming.show_matrix(hamming.G, "G")
# hamming.show_matrix(hamming.H, "H")
# message = [1, 0, 1, 0]
# message_encode = hamming.encode(message)
# # error = hamming.generate_error_vector(0.1)
# print("Encode message \n")
# print(message_encode)
# hamming.decode(message_encode)
# print(error)
# end test (work with 7 4 3)
#
