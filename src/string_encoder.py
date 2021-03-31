#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random


class StringEncoder(object):
    def __init__(self, n_bits: int = 40, enc_size: int = 2048, seed=12345):
        self.encodings = {}
        self.bits = {}

        self.n_bits = n_bits
        self.enc_size = enc_size

        random.seed(seed)

    def encode(self, string):

        if string in self.encodings:
            enc = self.encodings[string]
        else:

            enc = set(random.sample(population=range(self.enc_size), k=self.n_bits))
            self.encodings[string] = enc

            # maintain the mapping of bits to bucket
            #
            for bit in enc:
                if bit not in self.bits:
                    self.bits[bit] = {string}
                else:
                    self.bits[bit].add(string)

        return enc

    def decode(self, enc):
        strings = {}

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

        # sum the weights for the buckets associated with the bits in the encoding
        #
        total_weight = 0.0
        for bit in enc:
            for string in self.bits[bit]:
                if string not in strings:
                    strings[string] = enc[bit]
                else:
                    strings[string] += enc[bit]
                total_weight += enc[bit]

        strings = {string: strings[string] / total_weight for string in strings}

        return strings


if __name__ == '__main__':

    from src.sdr import SDR

    encoder = StringEncoder(n_bits=40, enc_size=2048)

    sdr_1 = SDR(enc_type='test', value='hello', encoder=encoder)
    sdr_2 = SDR(enc_type='test', value='world', encoder=encoder)

    d = sdr_1.distance(sdr_2)
    o = sdr_1.overlap(sdr_2)

    sdr_3 = SDR()

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    sdr_3.learn(sdr_1, learn_rate=0.7, prune=0.01)

    sdr_4 = SDR()

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    sdr_3.learn(sdr_4, learn_rate=0.7, prune=0.01)

    print('finished')



