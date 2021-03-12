#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Dict, Tuple, Union, Set


class AMString(object):
    def __init__(self, string):
        self.string = []
        if isinstance(string, AMString):
            # make a copy
            self.string = [{ch: pos[ch] for ch in pos} for pos in string.string]
        else:
            for ch in string:
                # add in opposite case
                #
                if ch.islower():
                    self.string.append({ch: 1.0, ch.upper(): 0.5})
                else:
                    self.string.append({ch: 1.0, ch.lower(): 0.5})

    def distance_v1(self, string, normalise: Optional[int] = None) -> float:
        distance = 0.0

        if isinstance(string, AMString):
            str_to_compare = string.string
        else:
            str_to_compare =[]
            for ch in string:
                # add in opposite case
                #
                if ch.islower():
                    str_to_compare.append({ch: 1.0, ch.upper(): 0.5})
                else:
                    str_to_compare.append({ch: 1.0, ch.lower(): 0.5})

        string_len = max(len(self.string), len(str_to_compare))
        if normalise is None:
            normalise = string_len

        for pos in range(string_len):
            if pos < len(self.string) and pos < len(str_to_compare):
                chars_to_test = set(self.string[pos].keys()) | set(str_to_compare[pos].keys())
                for ch in chars_to_test:
                    if ch in self.string[pos] and ch in str_to_compare[pos]:
                        distance += pow(self.string[pos][ch] - str_to_compare[pos][ch], 2)
                    elif ch in self.string[pos]:
                        distance += pow(self.string[pos][ch], 2)
                    else:
                        distance += pow(str_to_compare[pos][ch], 2)
            elif pos < len(self.string):
                for ch in self.string[pos]:
                    distance += pow(self.string[pos][ch], 2)
            else:
                for ch in str_to_compare[pos]:
                    distance += pow(str_to_compare[pos][ch], 2)

        distance = pow(distance, 0.5) / pow(normalise * 4, 0.5)
        return distance

    def distance(self, string, normalise: Optional[int] = None) -> float:
        distance = 0.0

        if isinstance(string, AMString):
            str_to_compare = string.string
        else:
            str_to_compare =[]
            for ch in string:
                # add in opposite case
                #
                if ch.islower():
                    str_to_compare.append({ch: 1.0, ch.upper(): 0.5})
                else:
                    str_to_compare.append({ch: 1.0, ch.lower(): 0.5})

        string_len = max(len(self.string), len(str_to_compare))
        if normalise is None:
            normalise = string_len

        for pos in range(string_len):
            if pos < len(self.string) and pos < len(str_to_compare):
                dist = 0
                chars_to_test = set(self.string[pos].keys()) | set(str_to_compare[pos].keys())
                for ch in chars_to_test:
                    if ch in self.string[pos] and ch in str_to_compare[pos]:
                        dist += abs(self.string[pos][ch] - str_to_compare[pos][ch])
                    elif ch in self.string[pos]:
                        dist += self.string[pos][ch]
                    else:
                        dist += str_to_compare[pos][ch]
                distance = dist / len(chars_to_test)
            elif pos < len(self.string):
                dist = 0
                for ch in self.string[pos]:
                    dist += self.string[pos][ch]
                distance += dist / len(self.string[pos])
            else:
                dist = 0
                for ch in str_to_compare[pos]:
                    dist += str_to_compare[pos][ch]
                distance += dist / len(str_to_compare[pos])

        distance = distance / (normalise * 2)
        return distance

    def learn(self, string, learn_rate: float = 1.0, prune=0.001):

        if isinstance(string, AMString):
            str_to_learn = string.string
        else:
            str_to_learn = [{ch: 1.0} for ch in string]

        string_len = max(len(self.string), len(str_to_learn))

        pos_to_delete = []
        for pos in range(string_len):

            if pos < len(self.string) and pos < len(str_to_learn):
                chars_to_test = set(self.string[pos].keys()) | set(str_to_learn[pos].keys())
                for ch in chars_to_test:
                    if ch in self.string[pos] and ch in str_to_learn[pos]:
                        self.string[pos][ch] += (str_to_learn[pos][ch] - self.string[pos][ch]) * learn_rate
                    elif ch in self.string[pos]:
                        self.string[pos][ch] -= self.string[pos][ch] * learn_rate
                    else:
                        self.string[pos][ch] = str_to_learn[pos][ch] * learn_rate
                    if self.string[pos][ch] <= prune:
                        del self.string[pos][ch]

            elif pos < len(self.string):
                chars_to_test = list(self.string[pos].keys())
                for ch in chars_to_test:
                    self.string[pos][ch] -= self.string[pos][ch] * learn_rate
                    if self.string[pos][ch] <= prune:
                        del self.string[pos][ch]
            else:
                self.string.append({ch: str_to_learn[pos][ch] for ch in str_to_learn[pos]})

            if len(self.string[pos]) == 0:
                pos_to_delete.append(pos)
        if len(pos_to_delete):
            pos_to_delete.reverse()
            for pos in pos_to_delete:
                del self.string[pos]

    def decode(self, threshold=0.8) -> str:
        max_weight = max([pos[ch] for pos in self.string for ch in pos])
        cutoff = max_weight * threshold
        string = ''

        for pos in self.string:
            for ch in pos:
                if pos[ch] >= cutoff:
                    string = f'{string}{ch}'
        return string


if __name__ == '__main__':

    max_string_size = 100
    s0 = AMString('H')
    d0_1 = s0.distance('h', normalise=max_string_size)

    d0_2 = s0.distance('e', normalise=max_string_size)

    s1 = AMString("hello")
    d1 = s1.distance('Hello there')

    s2 = AMString(s1)
    print(s2.decode())

    s2.learn('Hello there')
    print(s2.decode())

    s2.learn('Hello', learn_rate=0.9)
    print(s2.decode())

    s2.learn('He', learn_rate=0.9)
    print(s2.decode())

    s2.learn('He', learn_rate=0.9)
    print(s2.decode())

    s2.learn('He', learn_rate=0.9)
    print(s2.decode())

    s2.learn('He', learn_rate=0.9)
    print(s2.decode())

    print('finished')