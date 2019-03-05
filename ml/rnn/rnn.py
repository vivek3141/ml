import numpy as np
from random import uniform


class CharRNN:
    def __init__(self):
        self.data = open('input.txt', 'r').read()  # should be simple plain text file
        self.chars = list(set(self.data))
        self.data_size, self.vocab_size = len(self.data), len(self.chars)
        print('self.data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        hidden_size = 100  # size of hidden layer of neurons
        self.seq_length = 25  # number of steps to unroll the RNN for
        self.learning_rate = 1e-1

        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((self.vocab_size, 1))  # output bias

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -np.log(ps[t][targets[t], 0])

        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[
                t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        :param h: Memory State
        :param seed_ix: Seed letter for first time stpe
        :param n: Number
        :return:
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def gradCheck(self, inputs, targets, hprev):
        num_checks, delta = 10, 1e-5
        _, dWxh, dWhh, dWhy, dbh, dby, _ = self.lossFun(inputs, targets, hprev)
        for param, dparam, name in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby],
                                       ['Wxh', 'Whh', 'Why', 'bh', 'by']):

            s0 = dparam.shape
            s1 = param.shape
            assert s0 == s1, 'Error dims dont match: %s and %s.' % (repr(s0), repr(s1))
            print(name)
            for i in range(num_checks):
                ri = int(uniform(0, param.size))
                # evaluate cost at [x + delta] and [x - delta]
                old_val = param.flat[ri]
                param.flat[ri] = old_val + delta
                cg0, _, _, _, _, _, _ = self.lossFun(inputs, targets, hprev)
                param.flat[ri] = old_val - delta
                cg1, _, _, _, _, _, _ = self.lossFun(inputs, targets, hprev)
                param.flat[ri] = old_val  # reset old value for this parameter
                # fetch both numerical and analytic gradient
                grad_analytic = dparam.flat[ri]
                grad_numerical = (cg0 - cg1) / (2 * delta)
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                # rel_error should be on order of 1e-7 or less

    def train(self):
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length  # loss at iteration 0
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + self.seq_length + 1 >= len(self.data) or n == 0:
                hprev = np.zeros((self.hidden_size, 1))  # reset RNN memory
                p = 0  # go from start of data
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p + self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + self.seq_length + 1]]

            # sample from the model now and then
            if n % 100 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print('----\n %s \n----' % (txt,))

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                          [dWxh, dWhh, dWhy, dbh, dby],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            p += self.seq_length  # move data pointer
            n += 1  # iteration counter

    # gradient checking


