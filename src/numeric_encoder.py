#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import random


class NumericEncoder(object):
    def __init__(self, min_step: float = 1.0, n_bits: int = 40, enc_size: int = 2048, seed=12345):
        self.encodings = {}
        self.bits = {}

        self.min_step = min_step
        self.n_bits = n_bits
        self.enc_size = enc_size

        self.min: int = 0
        self.max: int = 0
        self.zero_bucket = None
        random.seed(seed)

    def encode(self, numeric):

        # round the numeric to the minimum step
        #
        round_numeric = int(numeric / self.min_step) * self.min_step

        # if no existing encodings then create first one
        #
        if len(self.encodings) == 0:

            bucket = 0

            # setup up the zero bucket association with real number
            #
            self.zero_bucket = round_numeric

            enc = set(random.sample(population=range(self.enc_size), k=self.n_bits))
            self.encodings[bucket] = enc

            # the max and min bucket
            #
            self.max = bucket
            self.min = bucket

            # maintain the mapping of bits to bucket
            #
            for bit in enc:
                if bit not in self.bits:
                    self.bits[bit] = {bucket}
                else:
                    self.bits[bit].add(bucket)
        else:

            # calculate the bucket associated with the encoding
            #
            target_bucket = int((round_numeric - self.zero_bucket) / self.min_step)

            # just return encoding if bucket exists
            #
            if target_bucket in self.encodings:
                enc = self.encodings[target_bucket]

            elif target_bucket > self.max:
                # will need the bits from the current largest bucket encoding
                #
                prev_enc = self.encodings[self.max]
                for bucket in range(self.max + 1, target_bucket + 1):

                    # get another bit chosen at random
                    #
                    new_bit = random.sample(population=[i for i in range(self.enc_size) if i not in prev_enc], k=1)

                    # choose bit to replace
                    #
                    bit_to_replace = random.sample(population=prev_enc, k=1)

                    # create the new encoding and save it
                    #
                    new_enc = {i if i not in bit_to_replace else new_bit[0] for i in prev_enc}
                    self.encodings[bucket] = new_enc

                    # maintain the mapping of bits to buckets
                    #
                    for bit in new_enc:
                        if bit not in self.bits:
                            self.bits[bit] = {bucket}
                        else:
                            self.bits[bit].add(bucket)

                    # remember the previous encoding
                    #
                    prev_enc = new_enc

                # we now have a new max bucket
                #
                self.max = target_bucket

                # the required encoding
                #
                enc = self.encodings[target_bucket]

            else:
                prev_enc = self.encodings[self.min]
                for bucket in range(self.min - 1, target_bucket - 1, -1):

                    # get another bit chosen at random
                    #
                    new_bit = random.sample(population=[i for i in range(self.enc_size) if i not in prev_enc], k=1)

                    # choose bit to replace
                    #
                    bit_to_replace = random.sample(population=prev_enc, k=1)

                    # the new encoding
                    #
                    new_enc = {i if i not in bit_to_replace else new_bit[0] for i in prev_enc}
                    self.encodings[bucket] = new_enc

                    # maintain the mapping of bits to buckets
                    #
                    for bit in new_enc:
                        if bit not in self.bits:
                            self.bits[bit] = {bucket}
                        else:
                            self.bits[bit].add(bucket)

                    # remember the previous encoding
                    #
                    prev_enc = new_enc

                # we now have a new min bucket
                #
                self.min = target_bucket

                # the encoding required
                #
                enc = self.encodings[target_bucket]

        return enc

    def decode(self, enc):
        buckets = {}

        # add default weights if not given any
        #
        if isinstance(enc, set):
            enc = {bit: 1.0 for bit in enc}

        # sum the weights for the buckets associated with the bits in the encoding
        #
        for bit in enc:
            for bucket in self.bits[bit]:
                if bucket not in buckets:
                    buckets[bucket] = enc[bit]
                else:
                    buckets[bucket] += enc[bit]

        if len(buckets) > 0:
            # create a list of buckets so we can sort
            #
            buckets = [(n, buckets[n]) for n in buckets]
            buckets.sort(key=lambda x: x[1], reverse=True)

            best_weight = buckets[0][1]
            if best_weight < self.n_bits:
                value = 0.0
                total_weight = 0.0

                for idx in range(len(buckets)):
                    value += ((buckets[idx][0] * self.min_step) + self.zero_bucket) * buckets[idx][1]
                    total_weight += buckets[idx][1]

                value = round(value / total_weight / self.min_step) * self.min_step
            else:
                value = (buckets[0][0] * self.min_step) + self.zero_bucket
        else:
            value = None
        return value


if __name__ == '__main__':

    from src.sdr import SDR

    encoder = NumericEncoder(min_step=0.1, n_bits=4, enc_size=2048)

    enc_1 = encoder.encode(100)
    enc_4 = encoder.encode(120.0)

    val_1 = encoder.decode(enc_1)

    enc_2 = encoder.encode(102)

    val_2 = encoder.decode(enc_2)

    enc_3 = encoder.encode(100.5)
    val_3 = encoder.decode(enc_3)

    val_4 = encoder.decode(enc_4)

    sdr_1 = SDR(enc_type='test', value=100, encoder=encoder)
    sdr_2 = SDR(enc_type='test', value=102, encoder=encoder)
    sdr_3 = SDR(enc_type='test', value=100.5, encoder=encoder)

    d_1_2 = sdr_1.distance(sdr_2)

    d_1_3 = sdr_1.distance(sdr_3)

    sdr_1.learn(sdr=sdr_2, learn_rate=0.5, prune=0.01)
    val_1_learned_1 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=0.5, prune=0.01)

    val_1_learned_2 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=0.5, prune=0.01)

    val_1_learned_3 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=0.5, prune=0.01)

    val_1_learned_4 = sdr_1.decode()

    sdr_1.learn(sdr=sdr_2, learn_rate=0.5, prune=0.01)

    val_1_learned_5 = sdr_1.decode()

    print('finished')
