#!/usr/bin/env python
# -*- encoding: utf-8 -*-


class SDR(object):

    def __init__(self, enc_type=None, value=None, encoder=None):
        self.encodings = {}
        self.encoder = {}
        if enc_type is not None and value is not None and encoder is not None:
            self.add_encoding(enc_type=enc_type, value=value, encoder=encoder)

    def add_encoding(self, enc_type, value, encoder):
        self.encoder[enc_type] = encoder
        enc = encoder.encode(value)
        if isinstance(enc, set):
            self.encodings[enc_type] = {e: 1.0 for e in enc}
        elif isinstance(enc, dict):
            self.encodings[enc_type] = enc

    def get_enc_types(self):
        return set(self.encodings.keys())

    def decode(self):
        dec = {enc_type: self.encoder[enc_type].decode(self.encodings[enc_type]) for enc_type in self.encodings}
        return dec

    def overlap(self, sdr, search_types=None):

        # get the filtered intersection of enc_types to compare
        #
        if search_types is not None:
            enc_types = ({enc_type
                          for enc_type in self.encodings.keys()
                          if enc_type in search_types} &
                         {enc_type
                          for enc_type in sdr.encodings.keys()
                          if enc_type in search_types})
        else:
            enc_types = set(self.encodings.keys()) & set(sdr.encodings.keys())

        por = {'overlap': 0, 'norm_overlap': 0.0, 'enc_types': {}}

        for enc_type in enc_types:
            self_bits = set(self.encodings[enc_type].keys())
            sdr_bits = set(sdr.encodings[enc_type].keys())

            # sum the overlapping bit weights
            #
            por['enc_types'][enc_type] = sum([min(self.encodings[enc_type][b], sdr.encodings[enc_type][b]) for b in self_bits & sdr_bits])

            por['overlap'] += por['enc_types'][enc_type]

        if len(enc_types) > 0:
            por['norm_overlap'] = por['overlap'] / len(enc_types)

        return por

    def distance(self, sdr, search_types=None):

        if search_types is not None:
            enc_types = ({enc_type
                          for enc_type in self.encodings.keys()
                          if enc_type in search_types} |
                         {enc_type
                          for enc_type in sdr.encodings.keys()
                          if enc_type in search_types})
        else:
            enc_types = set(self.encodings.keys()) | set(sdr.encodings.keys())

        por = {'distance': 0, 'norm_distance': 0.0, 'enc_types': {}}

        for enc_type in enc_types:
            if enc_type in self.encodings and sdr.encodings:
                self_bits = set(self.encodings[enc_type].keys())
                sdr_bits = set(sdr.encodings[enc_type].keys())

                por['enc_types'][enc_type] = sum([abs(self.encodings[enc_type][b] - sdr.encodings[enc_type][b]) for b in self_bits & sdr_bits])
                por['enc_types'][enc_type] += sum([self.encodings[enc_type][b] for b in self_bits - sdr_bits])
                por['enc_types'][enc_type] += sum([sdr.encodings[enc_type][b] for b in sdr_bits - self_bits])

                por['distance'] += por['enc_types'][enc_type]
            elif enc_type in self.encodings:
                por['enc_types'][enc_type] = sum([self.encodings[enc_type][b] for b in self.encodings[enc_type]])
                por['distance'] += por['enc_types'][enc_type]
            else:
                por['enc_types'][enc_type] = sum([sdr.encodings[enc_type][b] for b in sdr.encodings[enc_type]])
                por['distance'] += por['enc_types'][enc_type]

        if len(enc_types) > 0:
            por['norm_distance'] = por['distance'] / len(enc_types)

        return por

    def learn(self, sdr, learn_rate=1.0, learn_types=None, prune=0.01):

        if learn_types is not None:
            enc_types = ({enc_type
                         for enc_type in self.encodings.keys()
                         if enc_type in learn_types} |
                         {enc_type
                          for enc_type in sdr.encodings.keys()
                          if enc_type in learn_types})

        else:
            enc_types = set(self.encodings.keys()) | set(sdr.encodings.keys())

        for enc_type in enc_types:
            if enc_type in self.encodings and sdr.encodings:
                self_bits = set(self.encodings[enc_type].keys())
                sdr_bits = set(sdr.encodings[enc_type].keys())
                bits = self_bits | sdr_bits
                for bit in bits:
                    if bit in self.encodings[enc_type] and bit in sdr.encodings[enc_type]:
                        self.encodings[enc_type][bit] += (sdr.encodings[enc_type][bit] - self.encodings[enc_type][bit]) * learn_rate
                    elif bit in self.encodings[enc_type]:
                        self.encodings[enc_type][bit] -= self.encodings[enc_type][bit] * learn_rate
                    else:
                        self.encodings[enc_type][bit] = sdr.encodings[enc_type][bit] * learn_rate

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]

            elif enc_type in self.encodings:
                for bit in list(self.encodings[enc_type]):
                    self.encodings[enc_type][bit] -= (self.encodings[enc_type][bit] * learn_rate)

                    # prune if required
                    #
                    if self.encodings[enc_type][bit] < prune:
                        del self.encodings[enc_type][bit]
            else:
                self.encodings[enc_type] = {bit: sdr.encodings[enc_type][bit] * learn_rate for bit in sdr.encodings[enc_type]}

            # prune the bit type if empty
            #
            if len(self.encodings[enc_type]) == 0:
                del self.encodings[enc_type]

        # make sure we have references to encoders
        #
        if len(self.encodings) > 0:
            for enc_type in sdr.encoder:
                self.encoder[enc_type] = sdr.encoder[enc_type]
        else:
            self.encoder = {}

    def merge(self, sdr, weight=1.0):
        for enc_type in sdr.encodings:
            if enc_type in self.encodings:
                for bit in sdr.encodings[enc_type].keys():
                    if bit in self.encodings[enc_type]:
                        self.encodings[enc_type][bit] += sdr.encodings[enc_type][bit] * weight
                    else:
                        self.encodings[enc_type][bit] = sdr.encodings[enc_type][bit] * weight
            else:
                self.encodings[enc_type] = {bit: sdr.encodings[enc_type][bit] * weight for bit in sdr.encodings[enc_type]}

        # make sure we have references to encoders
        #
        if len(self.encodings) > 0:
            for enc_type in sdr.encoder:
                self.encoder[enc_type] = sdr.encoder[enc_type]
        else:
            self.encoder = {}


if __name__ == '__main__':
    from src.numeric_encoder import NumericEncoder

    encoder = NumericEncoder(min_step=1,
                             n_bits=4,
                             enc_size=100,
                             seed=12345)

    sdr_1 = SDR(enc_type='volume', value=100, encoder=encoder)

    sdr_2 = SDR(enc_type='volume', value=110, encoder=encoder)

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
