#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""An implementation for converting unicode strings in Indian languages to WX-Roman 
and vice-versa."""

import re
import sys

__version__    = "1.0.0"
__license__    = "MIT"
__author__     = "Irshad Ahmad"
__maintainer__ = "Irshad Ahmad"
__credits__    = ["Irshad Ahmad", "Riyaz Ahmad", "Rashid Ahmad"]
__email__      = [
                 "irshad.bhat@research.iiit.ac.in",
                 "riyaz.bhat@research.iiit.ac.in",
                 "bhatirshad127@gmail.com"
                 ]

class wxilp():
    def __init__(self, order):
        if order == "utf2wx":
            self.initialize_utf2wx_hash()
        elif order == "wx2utf":
            self.initialize_wx2utf_hash()
        else:
            sys.stderr.write("EncodingError: invalid source/target encoding\n")
            sys.exit(0)
    
    def initialize_wx2utf_hash(self):
        # CONSONANTS
        self.hashc_w2i = {
                u"k":u"\xB3",
                u"K":u"\xB4", 
                u"g":u"\xB5", 
                u"G":u"\xB6",
                u"f":u"\xB7",
                u"c":u"\xB8",
                u"C":u"\xB9",
                u"j":u"\xBA",
                u"J":u"\xBB",
                u"F":u"\xBC",
                u"t":u"\xBD",
                u"T":u"\xBE",
                u"d":u"\xBF",
                u"D":u"\xC0",
                u"N":u"\xC1",
                u"w":u"\xC2",
                u"W":u"\xC3",
                u"x":u"\xC4",
                u"X":u"\xC5",
                u"n":u"\xC6",
                u"p":u"\xC8",
                u"P":u"\xC9",
                u"b":u"\xCA",
                u"B":u"\xCB",
                u"m":u"\xCC",
                u"y":u"\xCD",
                u"r":u"\xCF",
                u"l":u"\xD1",
                u"v":u"\xD4",
                u"S":u"\xD5",
                u"s":u"\xD7",
                u"R":u"\xD6",
                u"h":u"\xD8",
                u"_":u"\xE8",   
                u"Z":u"\xE9",
                u".":u"\xEA",
                u"Y":u"\xFB",
                u"lY":u"\xD2",
                #Added for tamil wx to utf -Rashid
                u"rY":u"\xD0",
                u"nY":u"\xC7",
                u"lYY":u"\xD3",
                }
        # VOWELS
        self.hashv_w2i = {
                u"a":u"\xA4",
                u"A":u"\xA5",
                u"aA":u"\xA5",
                u"i":u"\xA6",
                u"ai":u"\xA6",
                u"I":u"\xA7",
                u"aI":u"\xA7",
                u"u":u"\xA8",
                u"au":u"\xA8",
                u"U":u"\xA9",
                u"aU":u"\xA9",
                u"q":u"\xAA",
                u"aq":u"\xAA",
                u"eV":u"\xAB",
                u"aeV":u"\xAB",
                u"e":u"\xAC",
                u"ae":u"\xAC",
                u"E":u"\xAD",
                u"aE":u"\xAD",
                u"EY":u"\xAE",
                u"aEY":u"\xAE",
                u"oV":u"\xAF",
                u"aoV":u"\xAF",
                u"o":u"\xB0",
                u"ao":u"\xB0",
                u"O":u"\xB1",
                u"aO":u"\xB1",
                u"OY":u"\xB2",
                u"aOY":u"\xB2",
                }
        # MATRA
        self.hashm_w2i = {
                u"A":u"\xDA",
                u"aA":u"\xDA",
                u"i":u"\xDB",
                u"ai":u"\xDB",
                u"I":u"\xDC",
                u"aI":u"\xDC",
                u"u":u"\xDD",
                u"au":u"\xDD",
                u"U":u"\xDE",
                u"aU":u"\xDE",
                u"q":u"\xDF",
                u"aq":u"\xDF",
                u"eV":u"\xE0",
                u"aeV":u"\xE0",
                u"e":u"\xE1",
                u"ae":u"\xE1",
                u"E":u"\xE2",
                u"aE":u"\xE2",
                u"EY":u"\xE3",
                u"aEY":u"\xE3",
                u"oV":u"\xE4",
                u"aoV":u"\xE4",
                u"o":u"\xE5",
                u"ao":u"\xE5",
                u"O":u"\xE6",
                u"aO":u"\xE6",
                u"OY":u"\xE7",
                u"aOY":u"\xE7",
                }
        # MODIFIERS
        self.hashmd_w2i = {
                u"z":u"\xA1",
                u"M":u"\xA2",
                u"H":u"\xA3",
                }
        self.hashh_i2u = {
                u"\xA1":u"\u0901",      #Vowel-modifier CHANDRABINDU
                u"\xA2":u"\u0902",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0903",      #Vowel-modifier VISARG
                u"\xA4":u"\u0905",      #Vowel A
                u"\xA5":u"\u0906",      #Vowel AA
                u"\xA6":u"\u0907",      #Vowel I
                u"\xA7":u"\u0908",      #Vowel II
                u"\xA8":u"\u0909",      #Vowel U
                u"\xA9":u"\u090A",      #Vowel UU
                u"\xAA":u"\u090B",      #Vowel RI
                u"\xAB":u"\u090E",      #Vowel E
                u"\xAC":u"\u090F",      #Vowel EY
                u"\xAD":u"\u0910",      #Vowel AI
                u"\xAE":u"\u090D",      #Vowel AI -Rashid added
                u"\xB2":u"\u090D",      #Vowel AYE (Devanagari Script)
                u"\xAF":u"\u0912",      #Vowel O
                u"\xB0":u"\u0913",      #Vowel OW
                u"\xB1":u"\u0914",      #Vowel AU
                u"\xB2":u"\u0911",      #Vowel AWE
                u"\xB3":u"\u0915",      #Consonant KA
                u"\xB4":u"\u0916",      #Consonant KHA
                u"\xB5":u"\u0917",      #Consonant GA
                u"\xB6":u"\u0918",      #Consonant GHA
                u"\xB7":u"\u0919",      #Consonant NGA
                u"\xB8":u"\u091A",      #Consonant CHA
                u"\xB9":u"\u091B",      #Consonant CHHA
                u"\xBA":u"\u091C",      #Consonant JA
                u"\xBB":u"\u091D",      #Consonant JHA
                u"\xBC":u"\u091E",      #Consonant JNA
                u"\xBD":u"\u091F",      #Consonant Hard TA
                u"\xBE":u"\u0920",      #Consonant Hard THA
                u"\xBF":u"\u0921",      #Consonant Hard DA
                u"\xC0":u"\u0922",      #Consonant Hard DHA
                u"\xC1":u"\u0923",      #Consonant Hard NA
                u"\xC2":u"\u0924",      #Consonant Soft TA
                u"\xC3":u"\u0925",      #Consonant Soft THA
                u"\xC4":u"\u0926",      #Consonant Soft DA
                u"\xC5":u"\u0927",      #Consonant Soft DHA
                u"\xC6":u"\u0928",      #Consonant Soft NA
                u"\xC7":u"\u0929",      #Consonant NA (Tamil)
                u"\xC8":u"\u092A",      #Consonant PA
                u"\xC9":u"\u092B",      #Consonant PHA
                u"\xCA":u"\u092C",      #Consonant BA
                u"\xCB":u"\u092D",      #Consonant BHA
                u"\xCC":u"\u092E",      #Consonant MA
                u"\xCD":u"\u092F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0930",      #Consonant RA
                u"\xD0":u"\u0931",      #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0932",      #Consonant LA
                u"\xD2":u"\u0933",      #Consonant Hard LA
                u"\xD3":u"\u0934",      #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0935",      #Consonant VA
                u"\xD5":u"\u0936",      #Consonant SHA
                u"\xD6":u"\u0937",      #Consonant Hard SHA
                u"\xD7":u"\u0938",      #Consonant SA
                u"\xD8":u"\u0939",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u093E",      #Vowel Sign AA
                u"\xDB":u"\u093F",      #Vowel Sign I
                u"\xDC":u"\u0940",      #Vowel Sign II
                u"\xDD":u"\u0941",      #Vowel Sign U
                u"\xDE":u"\u0942",      #Vowel Sign UU
                u"\xDF":u"\u0943",      #Vowel Sign RI
                u"\xE0":u"\u0946",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0947",      #Vowel Sign EY
                u"\xE2":u"\u0948",      #Vowel Sign AI
                u"\xE3":u"\u0945",      #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u094A",      #Vowel Sign O
                u"\xE5":u"\u094B",      #Vowel Sign OW
                u"\xE6":u"\u094C",      #Vowel Sign AU
                u"\xE7":u"\u0949",      #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u094D",      #Vowel Omission Sign (Halant)
                u"\xE9":u"\u093C",      #Diacritic Sign (Nukta)
                u"\xEA":u"\u0964",      #Full Stop (Viram, Northern Scripts)
                #u"\xEA":u".",          #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0966",      #Digit 0
                u"\xF2":u"\u0967",      #Digit 1
                u"\xF3":u"\u0968",      #Digit 2
                u"\xF4":u"\u0969",      #Digit 3
                u"\xF5":u"\u096A",      #Digit 4
                u"\xF6":u"\u096B",      #Digit 5
                u"\xF7":u"\u096C",      #Digit 6
                u"\xF8":u"\u096D",      #Digit 7
                u"\xF9":u"\u096E",      #Digit 8
                u"\xFA":u"\u096F",      #Digit 9 
               }
        self.hasht_i2u = {
                u"\xA1":u"\u0C01",      #Vowel-modifier CHANDRABINDU
                u"\xA2":u"\u0C02",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0C03",      #Vowel-modifier VISARG
                u"\xA4":u"\u0C05",      #Vowel A
                u"\xA5":u"\u0C06",      #Vowel AA
                u"\xA6":u"\u0C07",      #Vowel I
                u"\xA7":u"\u0C08",      #Vowel II
                u"\xA8":u"\u0C09",      #Vowel U
                u"\xA9":u"\u0C0A",      #Vowel UU
                u"\xAA":u"\u0C0B",      #Vowel RI
                u"\xAB":u"\u0C0E",      #Vowel E
                u"\xAC":u"\u0C0F",      #Vowel EY
                u"\xAD":u"\u0C10",      #Vowel AI
                #u"\xB2":u"\u090D",     #Vowel AYE (Devanagari Script) 7-Mar-11 -Rashid
                u"\xB2":u"",            #Vowel AYE (Devanagari Script)
                u"\xAF":u"\u0C12",      #Vowel O
                u"\xB0":u"\u0C13",      #Vowel OW
                u"\xB1":u"\u0C14",      #Vowel AU
                #u"\xB2":u"\u0911",     #Vowel AWE 7-Mar-11 -Rashid
                u"\xB2":u"",            #Vowel AWE
                u"\xB3":u"\u0C15",      #Consonant KA
                u"\xB4":u"\u0C16",      #Consonant KHA
                u"\xB5":u"\u0C17",      #Consonant GA
                u"\xB6":u"\u0C18",      #Consonant GHA
                u"\xB7":u"\u0C19",      #Consonant NGA
                u"\xB8":u"\u0C1A",      #Consonant CHA
                u"\xB9":u"\u0C1B",      #Consonant CHHA
                u"\xBA":u"\u0C1C",      #Consonant JA
                u"\xBB":u"\u0C1D",      #Consonant JHA
                u"\xBC":u"\u0C1E",      #Consonant JNA
                u"\xBD":u"\u0C1F",      #Consonant Hard TA
                u"\xBE":u"\u0C20",      #Consonant Hard THA
                u"\xBF":u"\u0C21",      #Consonant Hard DA
                u"\xC0":u"\u0C22",      #Consonant Hard DHA
                u"\xC1":u"\u0C23",      #Consonant Hard NA
                u"\xC2":u"\u0C24",      #Consonant Soft TA
                u"\xC3":u"\u0C25",      #Consonant Soft THA
                u"\xC4":u"\u0C26",      #Consonant Soft DA
                u"\xC5":u"\u0C27",      #Consonant Soft DHA
                u"\xC6":u"\u0C28",      #Consonant Soft NA
                #u"\xC7":u"\u0929",     #Consonant NA (Tamil) 28-Feb-11 -Rashid
                u"\xC7":u"",            #Consonant NA (Tamil)
                u"\xC8":u"\u0C2A",      #Consonant PA
                u"\xC9":u"\u0C2B",      #Consonant PHA
                u"\xCA":u"\u0C2C",      #Consonant BA
                u"\xCB":u"\u0C2D",      #Consonant BHA
                u"\xCC":u"\u0C2E",      #Consonant MA
                u"\xCD":u"\u0C2F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0C30",      #Consonant RA
                u"\xD0":u"\u0C31",      #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0C32",      #Consonant LA
                u"\xD2":u"\u0C33",      #Consonant Hard LA
                #u"\xD3":u"\u0934",     #Consonant LLLA 7-Mar-11 -Rashid
                u"\xD3":u"",            #Consonant LLLA
                u"\xD4":u"\u0C35",      #Consonant VA
                u"\xD5":u"\u0C36",      #Consonant SHA
                u"\xD6":u"\u0C37",      #Consonant Hard SHA
                u"\xD7":u"\u0C38",      #Consonant SA
                u"\xD8":u"\u0C39",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0C3E",      #Vowel Sign AA
                u"\xDB":u"\u0C3F",      #Vowel Sign I
                u"\xDC":u"\u0C40",      #Vowel Sign II
                u"\xDD":u"\u0C41",      #Vowel Sign U
                u"\xDE":u"\u0C42",      #Vowel Sign UU
                u"\xDF":u"\u0C43",      #Vowel Sign RI
                u"\xE0":u"\u0C46",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0C47",      #Vowel Sign EY
                u"\xE2":u"\u0C48",      #Vowel Sign AI
                #u"\xE3":u"\u0945",     #Vowel Sign AYE (Devanagari Script) 7-Mar-11 -Rashid
                u"\xE3":u"",            #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u0C4A",      #Vowel Sign O
                u"\xE5":u"\u0C4B",      #Vowel Sign OW
                u"\xE6":u"\u0C4C",      #Vowel Sign AU
                #u"\xE7":u"\u0949",     #Vowel Sign AWE (Devanagari Script) 7-Mar-11 -Rashid
                u"\xE7":u"",            #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0C4D",      #Vowel Omission Sign (Halant)
                #u"\xE9":u"\u093C",     #Diacritic Sign (Nukta) 7-Mar-11 -Rashid
                u"\xE9":u"",            #Diacritic Sign (Nukta)
                u"\xEA":u".",           #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0C66",      #Digit 0
                u"\xF2":u"\u0C67",      #Digit 1
                u"\xF3":u"\u0C68",      #Digit 2
                u"\xF4":u"\u0C69",      #Digit 3
                u"\xF5":u"\u0C6A",      #Digit 4
                u"\xF6":u"\u0C6B",      #Digit 5
                u"\xF7":u"\u0C6C",      #Digit 6
                u"\xF8":u"\u0C6D",      #Digit 7
                u"\xF9":u"\u0C6E",      #Digit 8
                u"\xFA":u"\u0C6F",      #Digit 9 
                }
        self.hashp_i2u = {
                u"\xA1":u"\u0A01",      #Vowel-modifier CHANDRABINDU
                u"\xA2":u"\u0A02",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0A03",      #Vowel-modifier VISARG
                u"\xA4":u"\u0A05",      #Vowel A
                u"\xA5":u"\u0A06",      #Vowel AA
                u"\xA6":u"\u0A07",      #Vowel I
                u"\xA7":u"\u0A08",      #Vowel II
                u"\xA8":u"\u0A09",      #Vowel U
                u"\xA9":u"\u0A0A",      #Vowel UU
                u"\xAA":u"\u0A0B",      #Vowel RI
                u"\xAB":u"\u0A0E",      #Vowel E
                u"\xAC":u"\u0A0F",      #Vowel EY
                u"\xAD":u"\u0A10",      #Vowel AI
                u"\xB2":u"\u0A0D",      #Vowel AYE (Devanagari Script)
                u"\xAF":u"\u0A12",      #Vowel O
                u"\xB0":u"\u0A13",      #Vowel OW
                u"\xB1":u"\u0A14",      #Vowel AU
                u"\xB2":u"\u0A11",      #Vowel AWE
                u"\xB3":u"\u0A15",      #Consonant KA
                u"\xB4":u"\u0A16",      #Consonant KHA
                u"\xB5":u"\u0A17",      #Consonant GA
                u"\xB6":u"\u0A18",      #Consonant GHA
                u"\xB7":u"\u0A19",      #Consonant NGA
                u"\xB8":u"\u0A1A",      #Consonant CHA
                u"\xB9":u"\u0A1B",      #Consonant CHHA
                u"\xBA":u"\u0A1C",      #Consonant JA
                u"\xBB":u"\u0A1D",      #Consonant JHA
                u"\xBC":u"\u0A1E",      #Consonant JNA
                u"\xBD":u"\u0A1F",      #Consonant Hard TA
                u"\xBE":u"\u0A20",      #Consonant Hard THA
                u"\xBF":u"\u0A21",      #Consonant Hard DA
                u"\xC0":u"\u0A22",      #Consonant Hard DHA
                u"\xC1":u"\u0A23",      #Consonant Hard NA
                u"\xC2":u"\u0A24",      #Consonant Soft TA
                u"\xC3":u"\u0A25",      #Consonant Soft THA
                u"\xC4":u"\u0A26",      #Consonant Soft DA
                u"\xC5":u"\u0A27",      #Consonant Soft DHA
                u"\xC6":u"\u0A28",      #Consonant Soft NA
                u"\xC7":u"\u0A29",      #Consonant NA (Tamil)
                u"\xC8":u"\u0A2A",      #Consonant PA
                u"\xC9":u"\u0A2B",      #Consonant PHA
                u"\xCA":u"\u0A2C",      #Consonant BA
                u"\xCB":u"\u0A2D",      #Consonant BHA
                u"\xCC":u"\u0A2E",      #Consonant MA
                u"\xCD":u"\u0A2F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0A30",      #Consonant RA
                u"\xD0":u"\u0A31",      #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0A32",      #Consonant LA
                u"\xD2":u"\u0A33",      #Consonant Hard LA
                u"\xD3":u"\u0A34",      #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0A35",      #Consonant VA
                u"\xD5":u"\u0A36",      #Consonant SHA
                u"\xD6":u"\u0A37",      #Consonant Hard SHA
                u"\xD7":u"\u0A38",      #Consonant SA
                u"\xD8":u"\u0A39",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0A3E",      #Vowel Sign AA
                u"\xDB":u"\u0A3F",      #Vowel Sign I
                u"\xDC":u"\u0A40",      #Vowel Sign II
                u"\xDD":u"\u0A41",      #Vowel Sign U
                u"\xDE":u"\u0A42",      #Vowel Sign UU
                u"\xDF":u"\u0A43",      #Vowel Sign RI
                u"\xE0":u"\u0A46",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0A47",      #Vowel Sign EY
                u"\xE2":u"\u0A48",      #Vowel Sign AI
                u"\xE3":u"\u0A45",      #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u0A4A",      #Vowel Sign O
                u"\xE5":u"\u0A4B",      #Vowel Sign OW
                u"\xE6":u"\u0A4C",      #Vowel Sign AU
                u"\xE7":u"\u0A49",      #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0A4D",      #Vowel Omission Sign (Halant)
                u"\xE9":u"\u0A3C",      #Diacritic Sign (Nukta)
                u"\xEA":u"\u0964",      #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0A66",      #Digit 0
                u"\xF2":u"\u0A67",      #Digit 1
                u"\xF3":u"\u0A68",      #Digit 2
                u"\xF4":u"\u0A69",      #Digit 3
                u"\xF5":u"\u0A6A",      #Digit 4
                u"\xF6":u"\u0A6B",      #Digit 5
                u"\xF7":u"\u0A6C",      #Digit 6
                u"\xF8":u"\u0A6D",      #Digit 7
                u"\xF9":u"\u0A6E",      #Digit 8
                u"\xFA":u"\u0A6F",      #Digit 9 
                } 
        self.hashk_i2u = {
                u"\xA2":u"\u0C82",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0C83",      #Vowel-modifier VISARG
                u"\xA4":u"\u0C85",      #Vowel A
                u"\xA5":u"\u0C86",      #Vowel AA
                u"\xA6":u"\u0C87",      #Vowel I
                u"\xA7":u"\u0C88",      #Vowel II
                u"\xA8":u"\u0C89",      #Vowel U
                u"\xA9":u"\u0C8A",      #Vowel UU
                u"\xAA":u"\u0C8B",      #Vowel RI
                u"\xAB":u"\u0C0E",      #Vowel E
                u"\xAC":u"\u0C0F",      #Vowel EY
                u"\xAD":u"\u0C10",      #Vowel AI
                u"\xB2":u"\u0C0D",      #Vowel AYE (Devanagari Script)
                u"\xAF":u"\u0C12",      #Vowel O
                u"\xB0":u"\u0C13",      #Vowel OW
                u"\xB1":u"\u0C14",      #Vowel AU
                u"\xB2":u"\u0C11",      #Vowel AWE
                u"\xB3":u"\u0C15",      #Consonant KA
                u"\xB4":u"\u0C16",      #Consonant KHA
                u"\xB5":u"\u0C17",      #Consonant GA
                u"\xB6":u"\u0C18",      #Consonant GHA
                u"\xB7":u"\u0C19",      #Consonant NGA
                u"\xB8":u"\u0C1A",      #Consonant CHA
                u"\xB9":u"\u0C1B",      #Consonant CHHA
                u"\xBA":u"\u0C1C",      #Consonant JA
                u"\xBB":u"\u0C1D",      #Consonant JHA
                u"\xBC":u"\u0C1E",      #Consonant JNA
                u"\xBD":u"\u0C1F",      #Consonant Hard TA
                u"\xBE":u"\u0C20",      #Consonant Hard THA
                u"\xBF":u"\u0C21",      #Consonant Hard DA
                u"\xC0":u"\u0C22",      #Consonant Hard DHA
                u"\xC1":u"\u0C23",      #Consonant Hard NA
                u"\xC2":u"\u0C24",      #Consonant Soft TA
                u"\xC3":u"\u0C25",      #Consonant Soft THA
                u"\xC4":u"\u0C26",      #Consonant Soft DA
                u"\xC5":u"\u0C27",      #Consonant Soft DHA
                u"\xC6":u"\u0C28",      #Consonant Soft NA
                u"\xC7":u"\u0C29",      #Consonant NA (Tamil)
                u"\xC8":u"\u0C2A",      #Consonant PA
                u"\xC9":u"\u0C2B",      #Consonant PHA
                u"\xCA":u"\u0C2C",      #Consonant BA
                u"\xCB":u"\u0C2D",      #Consonant BHA
                u"\xCC":u"\u0C2E",      #Consonant MA
                u"\xCD":u"\u0C2F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0C30",      #Consonant RA
                u"\xD0":u"\u0C31",      #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0C32",      #Consonant LA
                u"\xD2":u"\u0C33",      #Consonant Hard LA
                u"\xD3":u"\u0C34",      #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0C35",      #Consonant VA
                u"\xD5":u"\u0C36",      #Consonant SHA
                u"\xD6":u"\u0C37",      #Consonant Hard SHA
                u"\xD7":u"\u0C38",      #Consonant SA
                u"\xD8":u"\u0C39",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0C3E",      #Vowel Sign AA
                u"\xDB":u"\u0C3F",      #Vowel Sign I
                u"\xDC":u"\u0C40",      #Vowel Sign II
                u"\xDD":u"\u0C41",      #Vowel Sign U
                u"\xDE":u"\u0C42",      #Vowel Sign UU
                u"\xDF":u"\u0C43",      #Vowel Sign RI
                u"\xE0":u"\u0C46",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0C47",      #Vowel Sign EY
                u"\xE2":u"\u0C48",      #Vowel Sign AI
                u"\xE3":u"\u0C45",      #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u0C4A",      #Vowel Sign O
                u"\xE5":u"\u0C4B",      #Vowel Sign OW
                u"\xE6":u"\u0C4C",      #Vowel Sign AU
                u"\xE7":u"\u0C49",      #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0C4D",      #Vowel Omission Sign (Halant)
                u"\xE9":u"\u0C3C",      #Diacritic Sign (Nukta)
                #u"\xEA":u"\u0C64",     #Full Stop (Viram, Northern Scripts)
                u"\xEA":u".",           #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0C66",      #Digit 0
                u"\xF2":u"\u0C67",      #Digit 1
                u"\xF3":u"\u0C68",      #Digit 2
                u"\xF4":u"\u0C69",      #Digit 3
                u"\xF5":u"\u0C6A",      #Digit 4
                u"\xF6":u"\u0C6B",      #Digit 5
                u"\xF7":u"\u0C6C",      #Digit 6
                u"\xF8":u"\u0C6D",      #Digit 7
                u"\xF9":u"\u0C6E",      #Digit 8
                u"\xFA":u"\u0C6F",      #Digit 9 
                }
        self.hashm_i2u = {
                u"\xA2":u"\u0D02",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0D03",      #Vowel-modifier VISARG
                u"\xA4":u"\u0D05",      #Vowel A
                u"\xA5":u"\u0D06",      #Vowel AA
                u"\xA6":u"\u0D07",      #Vowel I
                u"\xA7":u"\u0D08",      #Vowel II
                u"\xA8":u"\u0D09",      #Vowel U
                u"\xA9":u"\u0D0A",      #Vowel UU
                u"\xAA":u"\u0D0B",      #Vowel RI
                u"\xAB":u"\u0D0E",      #Vowel E
                u"\xAC":u"\u0D0F",      #Vowel EY
                u"\xAD":u"\u0D10",      #Vowel AI
                u"\xAF":u"\u0D12",      #Vowel O
                u"\xB0":u"\u0D13",      #Vowel OW
                u"\xB1":u"\u0D14",      #Vowel AU
                u"\xB2":u"\u0D11",      #Vowel AWE
                u"\xB3":u"\u0D15",      #Consonant KA
                u"\xB4":u"\u0D16",      #Consonant KHA
                u"\xB5":u"\u0D17",      #Consonant GA
                u"\xB6":u"\u0D18",      #Consonant GHA
                u"\xB7":u"\u0D19",      #Consonant NGA
                u"\xB8":u"\u0D1A",      #Consonant CHA
                u"\xB9":u"\u0D1B",      #Consonant CHHA
                u"\xBA":u"\u0D1C",      #Consonant JA
                u"\xBB":u"\u0D1D",      #Consonant JHA
                u"\xBC":u"\u0D1E",      #Consonant JNA
                u"\xBD":u"\u0D1F",      #Consonant Hard TA
                u"\xBE":u"\u0D20",      #Consonant Hard THA
                u"\xBF":u"\u0D21",      #Consonant Hard DA
                u"\xC0":u"\u0D22",      #Consonant Hard DHA
                u"\xC1":u"\u0D23",      #Consonant Hard NA
                u"\xC2":u"\u0D24",      #Consonant Soft TA
                u"\xC3":u"\u0D25",      #Consonant Soft THA
                u"\xC4":u"\u0D26",      #Consonant Soft DA
                u"\xC5":u"\u0D27",      #Consonant Soft DHA
                u"\xC6":u"\u0D28",      #Consonant Soft NA
                u"\xC7":u"\u0D29",      #Consonant NA (Tamil)
                u"\xC8":u"\u0D2A",      #Consonant PA
                u"\xC9":u"\u0D2B",      #Consonant PHA
                u"\xCA":u"\u0D2C",      #Consonant BA
                u"\xCB":u"\u0D2D",      #Consonant BHA
                u"\xCC":u"\u0D2E",      #Consonant MA
                u"\xCD":u"\u0D2F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0D30",      #Consonant RA
                u"\xD0":u"\u0D31",      #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0D32",      #Consonant LA
                u"\xD2":u"\u0D33",      #Consonant Hard LA
                u"\xD3":u"\u0D34",      #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0D35",      #Consonant VA
                u"\xD5":u"\u0D36",      #Consonant SHA
                u"\xD6":u"\u0D37",      #Consonant Hard SHA
                u"\xD7":u"\u0D38",      #Consonant SA
                u"\xD8":u"\u0D39",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0D3E",      #Vowel Sign AA
                u"\xDB":u"\u0D3F",      #Vowel Sign I
                u"\xDC":u"\u0D40",      #Vowel Sign II
                u"\xDD":u"\u0D41",      #Vowel Sign U
                u"\xDE":u"\u0D42",      #Vowel Sign UU
                u"\xDF":u"\u0D43",      #Vowel Sign RI
                u"\xE0":u"\u0D46",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0D47",      #Vowel Sign EY
                u"\xE2":u"\u0D48",      #Vowel Sign AI
                u"\xE3":u"",            #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u0D4A",      #Vowel Sign O
                u"\xE5":u"\u0D4B",      #Vowel Sign OW
                u"\xE6":u"\u0D4C",      #Vowel Sign AU
                u"\xE7":u"",            #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0D4D",      #Vowel Omission Sign (Halant)
                #u"\xEA":u"\u0D64",     #Full Stop (Viram, Northern Scripts)
                u"\xEA":u".",           #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0D66",      #Digit 0
                u"\xF2":u"\u0D67",      #Digit 1
                u"\xF3":u"\u0D68",      #Digit 2
                u"\xF4":u"\u0D69",      #Digit 3
                u"\xF5":u"\u0D6A",      #Digit 4
                u"\xF6":u"\u0D6B",      #Digit 5
                u"\xF7":u"\u0D6C",      #Digit 6
                u"\xF8":u"\u0D6D",      #Digit 7
                u"\xF9":u"\u0D6E",      #Digit 8
                u"\xFA":u"\u0D6F",      #Digit 9
                }   
        self.hashb_i2u = {
                u"\xA1":u"\u0981",
                u"\xA2":u"\u0982",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0983",      #Vowel-modifier VISARG
                u"\xA4":u"\u0985",      #Vowel A
                u"\xA5":u"\u0986",      #Vowel AA
                u"\xA6":u"\u0987",      #Vowel I
                u"\xA7":u"\u0988",      #Vowel II
                u"\xA8":u"\u0989",      #Vowel U
                u"\xA9":u"\u098A",      #Vowel UU
                u"\xAA":u"\u098B",      #Vowel RI
                u"\xAB":u"\u098F",      #Vowel 
                u"\xAD":u"\u0990",      #Vowel AI
                u"\xAF":u"\u0993",      #Vowel O
                u"\xB0":u"",            #Vowel OW
                u"\xB1":u"\u0994",      #Vowel AU
                u"\xB3":u"\u0995",      #Consonant KA
                u"\xB4":u"\u0996",      #Consonant KHA
                u"\xB5":u"\u0997",      #Consonant GA
                u"\xB6":u"\u0998",      #Consonant GHA
                u"\xB7":u"\u0999",      #Consonant NGA
                u"\xB8":u"\u099A",      #Consonant CHA
                u"\xB9":u"\u099B",      #Consonant CHHA
                u"\xBA":u"\u099C",      #Consonant JA
                u"\xBB":u"\u099D",      #Consonant JHA
                u"\xBC":u"\u099E",      #Consonant JNA
                u"\xBD":u"\u099F",      #Consonant Hard TA
                u"\xBE":u"\u09A0",      #Consonant Hard THA
                u"\xBF":u"\u09A1",      #Consonant Hard DA
                u"\xC0":u"\u09A2",      #Consonant Hard DHA
                u"\xC1":u"\u09A3",      #Consonant Hard NA
                u"\xC2":u"\u09A4",      #Consonant Soft TA
                u"\xC3":u"\u09A5",      #Consonant Soft THA
                u"\xC4":u"\u09A6",      #Consonant Soft DA
                u"\xC5":u"\u09A7",      #Consonant Soft DHA
                u"\xC6":u"\u09A8",      #Consonant Soft NA
                u"\xC7":u"",            #Consonant NA (Tamil)
                u"\xC8":u"\u09AA",      #Consonant PA
                u"\xC9":u"\u09AB",      #Consonant PHA
                u"\xCA":u"\u09AC",      #Consonant BA
                u"\xCB":u"\u09AD",      #Consonant BHA
                u"\xCC":u"\u09AE",      #Consonant MA
                u"\xCD":u"\u09AF",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u09B0",      #Consonant RA
                u"\xD0":u"",   
                u"\xD1":u"\u09B2",      #Consonant LA
                u"\xD2":u"",            #Consonant Hard LA
                u"\xD3":u"",            #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"",            #Consonant VA
                u"\xD5":u"\u09B6",      #Consonant SHA
                u"\xD6":u"\u09B7",      #Consonant Hard SHA
                u"\xD7":u"\u09B8",      #Consonant SA
                u"\xD8":u"\u09B9",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u09BE",      #Vowel Sign AA
                u"\xDB":u"\u09BF",      #Vowel Sign I
                u"\xDC":u"\u09C0",      #Vowel Sign II
                u"\xDD":u"\u09C1",      #Vowel Sign U
                u"\xDE":u"\u09C2",      #Vowel Sign UU
                u"\xDF":u"\u09C3",      #Vowel Sign RI
                u"\xE0":u"\u09C7",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"",            #Vowel Sign EY
                u"\xE2":u"\u09C8",      #Vowel Sign AI
                u"\xE3":u"",            #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u09CB",      #Vowel Sign O
                u"\xE5":u"",            #Vowel Sign OW
                u"\xE6":u"\u09CC",      #Vowel Sign AU
                u"\xE7":u"",            #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u09CD",      #Vowel Omission Sign (Halant)
                u"\xE9":u"\u09BC",
                u"\xEA":u"\u0964",      #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u09E6",      #Digit 0
                u"\xF2":u"\u09E7",      #Digit 1
                u"\xF3":u"\u09E8",      #Digit 2
                u"\xF4":u"\u09E9",      #Digit 3
                u"\xF5":u"\u09EA",      #Digit 4
                u"\xF6":u"\u09EB",      #Digit 5
                u"\xF7":u"\u09EC",      #Digit 6
                u"\xF8":u"\u09ED",      #Digit 7
                u"\xF9":u"\u09EE",      #Digit 8
                u"\xFA":u"\u09EF",      #Digit 9
                }    
        self.hashcta_i2u = {
                #u"\xA2":u"\u0B82",     #Vowel-modifier ANUSWAR commented 14-March-11 -Rashid
                u"\xA2":u"\u0BAE\u0BCD",        #Vowel-modifier ANUSWAR is m + halant
                u"\xA3":u"\u0B83",      #Vowel-modifier VISARG
                u"\xA4":u"\u0B85",      #Vowel A
                u"\xA5":u"\u0B86",      #Vowel AA
                u"\xA6":u"\u0B87",      #Vowel I
                u"\xA7":u"\u0B88",      #Vowel II
                u"\xA8":u"\u0B89",      #Vowel U
                u"\xA9":u"\u0B8A",      #Vowel UU
                u"\xAB":u"\u0B8E",      #Vowel 
                u"\xAC":u"\u0B8F",
                u"\xAD":u"\u0B90",      #Vowel AI
                u"\xAF":u"\u0B92",      #Vowel O
                u"\xB0":u"\u0B93",      #Vowel OW
                u"\xB1":u"\u0B94",      #Vowel AU
                u"\xB3":u"\u0B95",      #Consonant KA
                u"\xB4":u"",            #Consonant KHA
                u"\xB5":u"",            #Consonant GA
                u"\xB6":u"",            #Consonant GHA
                u"\xB7":u"\u0B99",      #Consonant NGA
                u"\xB8":u"\u0B9A",      #Consonant CHA
                u"\xB9":u"",            #Consonant CHHA
                u"\xBA":u"\u0B9C",      #Consonant JA
                u"\xBB":u"",            #Consonant JHA
                u"\xBC":u"\u0B9E",      #Consonant JNA
                u"\xBD":u"\u0B9F",      #Consonant Hard TA
                u"\xBE":u"",            #Consonant Hard THA
                u"\xBF":u"",            #Consonant Hard DA
                u"\xC0":u"",            #Consonant Hard DHA
                u"\xC1":u"\u0BA3",      #Consonant Hard NA
                u"\xC2":u"\u0BA4",      #Consonant Soft TA
                u"\xC3":u"",            #Consonant Soft THA
                u"\xC4":u"",            #Consonant Soft DA
                u"\xC5":u"",            #Consonant Soft DHA
                u"\xC6":u"\u0BA8",      #Consonant Soft NA
                u"\xC7":u"\u0BA9",      #Consonant NA (Tamil)
                u"\xC8":u"\u0BAA",      #Consonant PA
                u"\xC9":u"",            #Consonant PHA
                u"\xCA":u"",            #Consonant BA
                u"\xCB":u"",            #Consonant BHA
                u"\xCC":u"\u0BAE",      #Consonant MA
                u"\xCD":u"\u0BAF",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0BB0",      #Consonant RA
                u"\xD0":u"\u0BB1",   
                u"\xD1":u"\u0BB2",      #Consonant LA
                u"\xD2":u"\u0BB3",      #Consonant Hard LA
                u"\xD3":u"\u0BB4",      #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0BB5",      #Consonant VA
                #u"\xD5":u"\u0BB6",     #Consonant SHA commented because it have same sence of 0BB7 14-Mar-11
                u"\xD5":u"\u0BB7",      #Consonant SHA is nomore use in tamil
                u"\xD6":u"\u0BB7",      #Consonant Hard SHA
                u"\xD7":u"\u0BB8",      #Consonant SA
                u"\xD8":u"\u0BB9",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0BBE",      #Vowel Sign AA
                u"\xDB":u"\u0BBF",      #Vowel Sign I
                u"\xDC":u"\u0BC0",      #Vowel Sign II
                u"\xDD":u"\u0BC1",      #Vowel Sign U
                u"\xDE":u"\u0BC2",      #Vowel Sign UU
                u"\xDF":u"",            #Vowel Sign RI
                u"\xE0":u"\u0BC6",      #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0BC7",      #Vowel Sign EY
                u"\xE2":u"\u0BC8",      #Vowel Sign AI
                u"\xE3":u"",            #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"\u0BCA",      #Vowel Sign O
                u"\xE5":u"\u0BCB",      #Vowel Sign OW
                u"\xE6":u"\u0BCC",      #Vowel Sign AU
                u"\xE7":u"",            #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0BCD",      #Vowel Omission Sign (Halant)
                u"\xE9":u"",
                #u"\xEA":u"\u0BE4",     #Full Stop (Viram, Northern Scripts)
                u"\xEA":u".",           #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0BE6",      #Digit 0
                u"\xF2":u"\u0BE7",      #Digit 1
                u"\xF3":u"\u0BE8",      #Digit 2
                u"\xF4":u"\u0BE9",      #Digit 3
                u"\xF5":u"\u0BEA",      #Digit 4
                u"\xF6":u"\u0BEB",      #Digit 5
                u"\xF7":u"\u0BEC",      #Digit 6
                u"\xF8":u"\u0BED",      #Digit 7
                u"\xF9":u"\u0BEE",      #Digit 8
                u"\xFA":u"\u0BEF",      #Digit 9
                }     
        self.hasho_i2u = {
                u"\xA1":u"\u0B01",      #Vowel-modifier CHANDRABINDU
                u"\xA2":u"\u0B02",      #Vowel-modifier ANUSWAR
                u"\xA3":u"\u0B03",      #Vowel-modifier VISARG
                u"\xA4":u"\u0B05",      #Vowel A
                u"\xA5":u"\u0B06",      #Vowel AA
                u"\xA6":u"\u0B07",      #Vowel I
                u"\xA7":u"\u0B08",      #Vowel II
                u"\xA8":u"\u0B09",      #Vowel U
                u"\xA9":u"\u0B0A",      #Vowel UU
                u"\xAA":u"\u0B0B",      #Vowel RI
                #u"\xAB":u"\u090E",     #Vowel E
                u"\xAC":u"\u0B0F",      #Vowel EY
                u"\xAD":u"\u0B10",      #Vowel AI
                #u"\xB2":u"\u0911",     #Vowel AWE
                #u"\xAE":u"\u090D",     #Vowel AI -Rashid added
                #u"\xB2":u"\u090D",     #Vowel AYE (Devanagari Script)
                #u"\xAF":u"\u0912",     #Vowel O
                u"\xB0":u"\u0B13",      #Vowel OW
                u"\xB1":u"\u0B14",      #Vowel AU
                u"\xB3":u"\u0B15",      #Consonant KA
                u"\xB4":u"\u0B16",      #Consonant KHA
                u"\xB5":u"\u0B17",      #Consonant GA
                u"\xB6":u"\u0B18",      #Consonant GHA
                u"\xB7":u"\u0B19",      #Consonant NGA
                u"\xB8":u"\u0B1A",      #Consonant CHA
                u"\xB9":u"\u0B1B",      #Consonant CHHA
                u"\xBA":u"\u0B1C",      #Consonant JA
                u"\xBB":u"\u0B1D",      #Consonant JHA
                u"\xBC":u"\u0B1E",      #Consonant JNA
                u"\xBD":u"\u0B1F",      #Consonant Hard TA
                u"\xBE":u"\u0B20",      #Consonant Hard THA
                u"\xBF":u"\u0B21",      #Consonant Hard DA
                u"\xC0":u"\u0B22",      #Consonant Hard DHA
                u"\xC1":u"\u0B23",      #Consonant Hard NA
                u"\xC2":u"\u0B24",      #Consonant Soft TA
                u"\xC3":u"\u0B25",      #Consonant Soft THA
                u"\xC4":u"\u0B26",      #Consonant Soft DA
                u"\xC5":u"\u0B27",      #Consonant Soft DHA
                u"\xC6":u"\u0B28",      #Consonant Soft NA
                u"\xC7":u"",            #Consonant NA (Tamil)
                u"\xC8":u"\u0B2A",      #Consonant PA
                u"\xC9":u"\u0B2B",      #Consonant PHA
                u"\xCA":u"\u0B2C",      #Consonant BA
                u"\xCB":u"\u0B2D",      #Consonant BHA
                u"\xCC":u"\u0B2E",      #Consonant MA
                u"\xCD":u"\u0B2F",      #Consonant YA
                u"\xCE":u"",            #Consonant JYA (Bangla, Assamese & Orriya)
                u"\xCF":u"\u0B30",      #Consonant RA
                u"\xD0":u"",            #Consonant Hard RA (Southern Script)
                u"\xD1":u"\u0B32",      #Consonant LA
                u"\xD2":u"\u0B33",      #Consonant Hard LA
                u"\xD3":u"",            #Consonant ZHA (Tamil & Malyalam)
                u"\xD4":u"\u0B35",      #Consonant VA
                u"\xD5":u"\u0B36",      #Consonant SHA
                u"\xD6":u"\u0B37",      #Consonant Hard SHA
                u"\xD7":u"\u0B38",      #Consonant SA
                u"\xD8":u"\u0B39",      #Consonant HA
                u"\xD9":u"",            #Consonant INV
                u"\xDA":u"\u0B3E",      #Vowel Sign AA
                u"\xDB":u"\u0B3F",      #Vowel Sign I
                u"\xDC":u"\u0B40",      #Vowel Sign II
                u"\xDD":u"\u0B41",      #Vowel Sign U
                u"\xDE":u"\u0B42",      #Vowel Sign UU
                u"\xDF":u"\u0B43",      #Vowel Sign RI
                u"\xE0":u"",            #Vowel Sign E (Southern Scripts)
                u"\xE1":u"\u0B47",      #Vowel Sign EY
                u"\xE2":u"\u0B48",      #Vowel Sign AI
                u"\xE3":u"",            #Vowel Sign AYE (Devanagari Script)
                u"\xE4":u"",            #Vowel Sign O
                u"\xE5":u"\u0B4B",      #Vowel Sign OW
                u"\xE6":u"\u0B4C",      #Vowel Sign AU
                u"\xE7":u"",            #Vowel Sign AWE (Devanagari Script)
                u"\xE8":u"\u0B4D",      #Vowel Omission Sign (Halant)
                u"\xE9":u"\u0B3C",      #Diacritic Sign (Nukta)
                u"\xEA":u"\u0964",      #Full Stop (Viram, Northern Scripts)
                #u"\xEA":u".",          #Full Stop (Viram, Northern Scripts)
                u"\xF1":u"\u0B66",      #Digit 0
                u"\xF2":u"\u0B67",      #Digit 1
                u"\xF3":u"\u0B68",      #Digit 2
                u"\xF4":u"\u0B69",      #Digit 3
                u"\xF5":u"\u0B6A",      #Digit 4
                u"\xF6":u"\u0B6B",      #Digit 5
                u"\xF7":u"\u0B6C",      #Digit 6
                u"\xF8":u"\u0B6D",      #Digit 7
                u"\xF9":u"\u0B6E",      #Digit 8
                u"\xFA":u"\u0B6F",      #Digit 9 
                }

        # compile regexes
        const = 'kKgGfcCjJFtTdDNwWxXnpPbBmyrlvSsRh'
        self.ceVmd = re.compile(u"([%s])eV([MHz])" %const)
        self.cceVmd = re.compile(u"([%s])([%s])eV([MHz])" %(const, const))
        self.ccceVmd = re.compile(u"([%s])([%s])([%s])eV([MHz])" %(const, const, const))

        self.ceV = re.compile(u"([%s])eV" %const)
        self.cceV = re.compile(u"([%s])([%s])eV" %(const, const))
        self.ccceV = re.compile(u"([%s])([%s])([%s])eV" %(const, const, const))

        self.cEYmd = re.compile(u"([%s])EY([MHz])" %const)
        self.ccEYmd = re.compile(u"([%s])([%s])EY([MHz])" %(const, const))
        self.cccEYmd = re.compile(u"([%s])([%s])([%s])EY([MHz])" %(const, const, const))

        self.cEY = re.compile(u"([%s])EY" %const)
        self.ccEY = re.compile(u"([%s])([%s])EY" %(const, const))
        self.cccEY = re.compile(u"([%s])([%s])([%s])EY" %(const, const, const))

        self.cOYmd = re.compile(u"([%s])OY([MHz])" %const)
        self.ccOYmd = re.compile(u"([%s])([%s])OY([MHz])" %(const, const))
        self.cccOYmd = re.compile(u"([%s])([%s])([%s])OY([MHz])" %(const, const, const))

        self.coVmd = re.compile(u"([%s])oV([MHz])" %const)
        self.ccoVmd = re.compile(u"([%s])([%s])oV([MHz])" %(const, const))
        self.cccoVmd = re.compile(u"([%s])([%s])([%s])oV([MHz])" %(const, const, const))

        self.coV = re.compile(u"([%s])oV" %const)
        self.ccoV = re.compile(u"([%s])([%s])oV" %(const, const))
        self.cccoV = re.compile(u"([%s])([%s])([%s])oV" %(const, const, const))

        self.cOY = re.compile(u"([%s])OY" %const)
        self.cZOY = re.compile(u"([%s])ZOY" %const)  #NOTE consonant+ZOY case added
        self.ccOY = re.compile(u"([%s])([%s])OY" %(const, const))
        self.cccOY = re.compile(u"([%s])([%s])([%s])OY" %(const, const, const))

        self.cvmd = re.compile(u"([%s])([AiIuUeEoO])([MHz])" %const)
        self.cZvmd = re.compile(u"([%s])Z([AiIuUeEoO])([MHz])" %const)
        self.ccvmd = re.compile(u"([%s])([%s])([AiIuUeEoO])([MHz])" %(const, const))
        self.cccvmd = re.compile(u"([%s])([%s])([%s])([AiIuUeEoO])([MHz])" %(const, const, const))

        self.cv = re.compile(u"([%s])([AiIuUeEoO])" %const)
        self.cZv = re.compile(u"([%s])Z([AiIuUeEoO])" %const)
        self.ccv = re.compile(u"([%s])([%s])([AiIuUeEoO])" %(const, const))
        self.cccv = re.compile(u"([%s])([%s])([%s])([AiIuUeEoO])" %(const, const, const))

        self.camd = re.compile(u"([%s])a([MHz])" %const)
        self.cZamd = re.compile(u"([%s])Za([MHz])" %const)
        self.cZmd = re.compile(u"([%s])Z([MHz])" %const) #NOTE consonant+Z+[MHz] case added
        self.ccamd = re.compile(u"([%s])([%s])a([MHz])" %(const, const))
        self.cccamd = re.compile(u"([%s])([%s])([%s])a([MHz])" %(const, const, const))

        self.ca = re.compile(u"([%s])a" %const)
        self.cZa = re.compile(u"([%s])Za" %const)
        self.cYZa = re.compile(u"([%s])YZa" %const) #NOTE consonant+YZa case added
        self.cca = re.compile(u"([%s])([%s])a" %(const, const))
        self.ccca = re.compile(u"([%s])([%s])([%s])a" %(const, const, const))

        self.c = re.compile(u"([%s])" %const)
        self.cZ = re.compile(u"([%s])Z" %const)
        self.cc = re.compile(u"([%s])([%s])" %(const, const))
        self.ccc = re.compile(u"([%s])([%s])([%s])" %(const, const, const))

        self.aqmd = re.compile(u"aq([MHz])")
        self.cq = re.compile(u"([%s])q" %const)
        self.cqmd = re.compile(u"([%s])q([MHz])" %const)
        self.qmd = re.compile(u"q([MHz])")    #NOTE q+[MHz]

        self.i2u_h = re.compile(u'([\xA1-\xFA])')
        self.i2u_t = re.compile(u'([\xA1-\xFA])')
        self.i2u_p = re.compile(u'([\xA1-\xFA])')
        self.i2u_k = re.compile(u'([\xA1-\xFA])')
        self.i2u_m = re.compile(u'([\xA1-\xFA])')
        self.i2u_b = re.compile(u'([\xA1-\xFA])')
        self.i2u_o = re.compile(u'([\xA1-\xFA])')
        self.i2u_ta = re.compile(u'([\xA1-\xFA])')

        #NOTE Handle pre-present iscii characters
        self.iscii_num = dict(zip([chr(i) for i in range(161, 251)], ['@~%s~@'%i for i in range(0, 90)]))
        self.num_iscii = dict(zip(['@~%s~@'%i for i in range(0, 90)], [chr(i) for i in range(161, 251)]))
        self.isc = re.compile(u'([\xA1-\xFA])')
        self.num = re.compile(u'(%s)' %'|'.join(['@~%s~@'%i for i in range(0, 90)]))

    def initialize_utf2wx_hash(self):

        self.BYTE_ORDER_MARK = u'\uFEFF'
        self.BYTE_ORDER_MARK_2 = u'\uFFFE'
        self.WORD_JOINER = u'\u2060'
        self.SOFT_HYPHEN = u'\u00AD'

        self.ZERO_WIDTH_NON_JOINER = u'\u200C'
        self.ZERO_WIDTH_JOINER = u'\u200D'

        self.hashc_i2w = {
                u"\xB3":u"k",
                u"\xB4":u"K",
                u"\xB5":u"g",
                u"\xB6":u"G",
                u"\xB7":u"f",
                u"\xB8":u"c",
                u"\xB9":u"C",
                u"\xBA":u"j",
                u"\xBB":u"J",
                u"\xBC":u"F",
                u"\xBD":u"t",
                u"\xBE":u"T",
                u"\xBF":u"d",
                u"\xC0":u"D",
                u"\xC1":u"N",
                u"\xC2":u"w",
                u"\xC3":u"W",
                u"\xC4":u"x",
                u"\xC5":u"X",
                u"\xC6":u"n",
                u"\xC7":u"nY",  #Representation for Consonant NA (Tamil) ??? Refer to ISCII-91.pdf page-16 -Rashid
                u"\xC8":u"p",
                u"\xC9":u"P",
                u"\xCA":u"b",
                u"\xCB":u"B",
                u"\xCC":u"m",
                u"\xCD":u"y",
                u"\xCE":u"",    #Representation for Consonant JYA in Bangla & Oriya??
                u"\xCF":u"r",
                u"\xD0":u"rY",  #Representation for Consonant HARD RA (Southern Script) -Rashid
                u"\xD1":u"l",
                u"\xD2":u"lY",
                u"\xD3":u"lYY", #Representation for Consonant ZHA (Tamil & Malyalam) - Rashid
                u"\xD4":u"v",
                u"\xD5":u"S",
                u"\xD6":u"R",
                u"\xD7":u"s",
                u"\xD8":u"h",
                u"\xE9":u"Z",   #NUKTA
                }
        self.hashv_i2w = {
                u"\xA4":u"a",
                u"\xA5":u"A", 
                u"\xA6":u"i", 
                u"\xA7":u"I", 
                u"\xA8":u"u",
                u"\xA9":u"U",
                u"\xAA":u"q",
                u"\xAB":u"eV",
                u"\xAC":u"e",
                u"\xAD":u"E",
                u"\xAE":u"EY",
                u"\xAF":u"oV",
                u"\xB0":u"o",
                u"\xB1":u"O",
                u"\xB2":u"OY",
                }
        self.hashm_i2w = {
                u"\xDA":u"A",
                u"\xDB":u"i",
                u"\xDC":u"I",
                u"\xDD":u"u",
                u"\xDE":u"U",
                u"\xDF":u"q",
                u"\xE0":u"eV",
                u"\xE1":u"e",
                u"\xE2":u"E",
                u"\xE3":u"EY",
                u"\xE4":u"oV",
                u"\xE5":u"o",
                u"\xE6":u"O",
                u"\xE7":u"OY",
                }   
        self.hashmd_i2w = {
                u"\xA1":u"z",
                u"\xA2":u"M",
                u"\xA3":u"H",
                }
        self.digits_i2w = {
                u"\xF1":u"0",
                u"\xF2":u"1",
                u"\xF3":u"2",
                u"\xF4":u"3",
                u"\xF5":u"4",
                u"\xF6":u"5",
                u"\xF7":u"6",
                u"\xF8":u"7",
                u"\xF9":u"8",
                u"\xFA":u"9",
                }
        self.hashh_u2i = {
                u"\u0901":u"\xA1",      #Vowel-modifier CHANDRABINDU
                u"\u0902":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0903":u"\xA3",      #Vowel-modifier VISARG
                u"\u0904":u"",
                u"\u0905":u"\xA4",      #Vowel A
                u"\u0906":u"\xA5",      #Vowel AA
                u"\u0907":u"\xA6",      #Vowel I
                u"\u0908":u"\xA7",      #Vowel II
                u"\u0909":u"\xA8",      #Vowel U
                u"\u090A":u"\xA9",      #Vowel UU
                u"\u090B":u"\xAA",      #Vowel RI
                u"\u090C":u"",
                u"\u090D":u"\xAE",
                u"\u090E":u"\xAB",
                u"\u090F":u"\xAC",
                u"\u0910":u"\xAD",
                u"\u0911":u"\xB2",
                u"\u0912":u"\xAF",
                u"\u0913":u"\xB0",
                u"\u0914":u"\xB1",
                u"\u0915":u"\xB3",      #Consonant KA
                u"\u0916":u"\xB4",      #Consonant
                u"\u0917":u"\xB5",      #Consonant
                u"\u0918":u"\xB6",      #Consonant
                u"\u0919":u"\xB7",      #Consonant NGA
                u"\u091A":u"\xB8",      #Consonant
                u"\u091B":u"\xB9",      #Consonant
                u"\u091C":u"\xBA",      #Consonant
                u"\u091D":u"\xBB",      #Consonant
                u"\u091E":u"\xBC",      #Consonant JNA
                u"\u091F":u"\xBD",      #Consonant
                u"\u0920":u"\xBE",      #Consonant
                u"\u0921":u"\xBF",      #Consonant
                u"\u0922":u"\xC0",      #Consonant
                u"\u0923":u"\xC1",      #Consonant NA
                u"\u0924":u"\xC2",      #Consonant
                u"\u0925":u"\xC3",      #Consonant
                u"\u0926":u"\xC4",      #Consonant
                u"\u0927":u"\xC5",      #Consonant
                u"\u0928":u"\xC6",      #Consonant NA
                u"\u0929":u"\xC7",      #Consonant NNNA
                u"\u092A":u"\xC8",      #Consonant PA
                u"\u092B":u"\xC9",      #Consonant PHA
                u"\u092C":u"\xCA",      #Consonant BA
                u"\u092D":u"\xCB",      #Consonant BHA
                u"\u092E":u"\xCC",      #Consonant MA
                u"\u092F":u"\xCD",      #Consonant YA
                u"\u0930":u"\xCF",      #Consonant RA
                u"\u0931":u"\xD0",      #Consonant RRA
                u"\u0932":u"\xD1",      #Consonant LA
                u"\u0933":u"\xD2",      #Consonant LLA
                u"\u0934":u"\xD3",      #Consonant LLLA
                u"\u0935":u"\xD4",      #Consonant VA
                u"\u0936":u"\xD5",      #Consonant SHA
                u"\u0937":u"\xD6",      #Consonant SSA
                u"\u0938":u"\xD7",      #Consonant SA
                u"\u0939":u"\xD8",      #Consonant HA
                u"\u093A":u"",          #Consonant
                u"\u093B":u"",          #Consonant
                u"\u093C":u"\xE9",      #Consonant NUKTA
                u"\u093D":u"",          #Consonant AVAGRAHA
                u"\u093E":u"\xDA",      #Vowel Sign AA
                u"\u093F":u"\xDB",      #Vowel Sign I
                u"\u0940":u"\xDC",      #Vowel Sign II
                u"\u0941":u"\xDD",      #Vowel Sign U
                u"\u0942":u"\xDE",      #Vowel 
                u"\u0943":u"\xDF",      #Vowel
                u"\u0944":u"",          #Vowel
                u"\u0945":u"",          #Vowel
                u"\u0946":u"\xE0",      #Vowel
                u"\u0947":u"\xE1",      #Vowel
                u"\u0948":u"\xE2",      #Vowel
                u"\u0949":u"\xE7",      #Vowel
                u"\u094A":u"\xE4",      #Vowel
                u"\u094B":u"\xE5",      #Vowel
                u"\u094C":u"\xE6",      #Vowel
                u"\u094D":u"\xE8",      # Halant
                u"\u0950":u"",          #Consonant
                u"\u0964":u"\xEA",      #Consonant
                u"\u0960":u"\xAA",      #Vowel Sanskrit
                u"\u0966":u"\xF1",      #Devanagari Digit 0
                u"\u0967":u"\xF2",      #Devanagari Digit 1
                u"\u0968":u"\xF3",      #Devanagari Digit 2
                u"\u0969":u"\xF4",      #Devanagari Digit 3
                u"\u096A":u"\xF5",      #Devanagari Digit 4
                u"\u096B":u"\xF6",      #Devanagari Digit 5
                u"\u096C":u"\xF7",      #Devanagari Digit 6
                u"\u096D":u"\xF8",      #Devanagari Digit 7
                u"\u096E":u"\xF9",      #Devanagari Digit 8
                u"\u096F":u"\xFA",      #Devanagari Digit 9
                }
        self.unicode_norm_hashh_u2i = {
                u"\u0958":u"\u0915",
                u"\u0959":u"\u0916",
                u"\u095A":u"\u0917",
                u"\u095B":u"\u091C",
                u"\u095C":u"\u0921",
                u"\u095D":u"\u0922",
                u"\u095E":u"\u092B",
                u"\u095F":u"\u092F",
                }
        self.hasht_u2i = { 
                u"\u0C01":u"\xA1",
                u"\u0C02":u"\xA2",
                u"\u0C03":u"\xA3",
                u"\u0C04":u"",
                u"\u0C05":u"\xA4",
                u"\u0C06":u"\xA5",
                u"\u0C07":u"\xA6",
                u"\u0C08":u"\xA7",
                u"\u0C09":u"\xA8",
                u"\u0C0A":u"\xA9",
                u"\u0C0B":u"\xAA",
                u"\u0C0C":u"",
                u"\u0C0D":u"\xAE",
                u"\u0C0E":u"\xAB",
                u"\u0C0F":u"\xAC",
                u"\u0C10":u"\xAD",
                u"\u0C11":u"\xB2",
                u"\u0C12":u"\xAF",
                u"\u0C13":u"\xB0",
                u"\u0C14":u"\xB1",
                u"\u0C15":u"\xB3",
                u"\u0C16":u"\xB4",
                u"\u0C17":u"\xB5",
                u"\u0C18":u"\xB6",
                u"\u0C19":u"\xB7",
                u"\u0C1A":u"\xB8",
                u"\u0C1B":u"\xB9",
                u"\u0C1C":u"\xBA",
                u"\u0C1D":u"\xBB",
                u"\u0C1E":u"\xBC",
                u"\u0C1F":u"\xBD",
                u"\u0C20":u"\xBE",
                u"\u0C21":u"\xBF",
                u"\u0C22":u"\xC0",
                u"\u0C23":u"\xC1",
                u"\u0C24":u"\xC2",
                u"\u0C25":u"\xC3",
                u"\u0C26":u"\xC4",
                u"\u0C27":u"\xC5",
                u"\u0C28":u"\xC6",
                u"\u0C29":u"\xC7",
                u"\u0C2A":u"\xC8",
                u"\u0C2B":u"\xC9",
                u"\u0C2C":u"\xCA",
                u"\u0C2D":u"\xCB",
                u"\u0C2E":u"\xCC",
                u"\u0C2F":u"\xCD",
                u"\u0C30":u"\xCF",
                u"\u0C31":u"\xD0",
                u"\u0C32":u"\xD1",
                u"\u0C33":u"\xD2",
                u"\u0C34":u"\xD3",
                u"\u0C35":u"\xD4",
                u"\u0C36":u"\xD5",
                u"\u0C37":u"\xD6",
                u"\u0C38":u"\xD7",
                u"\u0C39":u"\xD8",
                u"\u0C3A":u"",
                u"\u0C3B":u"",
                u"\u0C3C":u"",
                u"\u0C3D":u"",
                u"\u0C3E":u"\xDA",
                u"\u0C3F":u"\xDB",
                u"\u0C40":u"\xDC",
                u"\u0C41":u"\xDD",
                u"\u0C42":u"\xDE",
                u"\u0C43":u"\xDF",
                u"\u0C44":u"",
                u"\u0C45":u"",
                u"\u0C46":u"\xE0",
                u"\u0C47":u"\xE1",
                u"\u0C48":u"\xE2",
                u"\u0C49":u"",
                u"\u0C4A":u"\xE4",
                u"\u0C4B":u"\xE5",
                u"\u0C4C":u"\xE6",
                u"\u0C4D":u"\xE8",
                u"\u0C50":u"",
                u"\u0C64":u"\xEA",
                u"\u0C66":u"\xF1",
                u"\u0C67":u"\xF2",
                u"\u0C68":u"\xF3",
                u"\u0C69":u"\xF4",
                u"\u0C6A":u"\xF5",
                u"\u0C6B":u"\xF6",
                u"\u0C6C":u"\xF7",
                u"\u0C6D":u"\xF8",
                u"\u0C6E":u"\xF9",
                u"\u0C6F":u"\xFA",
                }
        self.hashp_u2i = { 
                #u"\x0A01":u"\xA1",     #Vowel-modifier CHANDRABINDU -Rashid
                u"\u0A02":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0A04":u"",
                u"\u0A05":u"\xA4",      #Vowel A
                u"\u0A06":u"\xA5",      #Vowel AA
                u"\u0A07":u"\xA6",      #Vowel I
                u"\u0A08":u"\xA7",      #Vowel II
                u"\u0A09":u"\xA8",      #Vowel U
                u"\u0A0A":u"\xA9",      #Vowel UU
                u"\u0A0B":u"\xAA",      #Vowel RI
                u"\u0A0C":u"",
                u"\u0A0D":u"\xAE",
                u"\u0A0E":u"\xAB",
                u"\u0A0F":u"\xAC",
                u"\u0A10":u"\xAD",
                u"\u0A11":u"\xB2",
                u"\u0A12":u"\xAF",
                u"\u0A13":u"\xB0",
                u"\u0A14":u"\xB1",
                u"\u0A15":u"\xB3",      #Consonant KA
                u"\u0A16":u"\xB4",      #Consonant
                u"\u0A17":u"\xB5",      #Consonant
                u"\u0A18":u"\xB6",      #Consonant
                u"\u0A19":u"\xB7",      #Consonant NGA
                u"\u0A1A":u"\xB8",      #Consonant
                u"\u0A1B":u"\xB9",      #Consonant
                u"\u0A1C":u"\xBA",      #Consonant
                u"\u0A1D":u"\xBB",      #Consonant
                u"\u0A1E":u"\xBC",      #Consonant JNA
                u"\u0A1F":u"\xBD",      #Consonant
                u"\u0A20":u"\xBE",      #Consonant
                u"\u0A21":u"\xBF",      #Consonant
                u"\u0A22":u"\xC0",      #Consonant
                u"\u0A23":u"\xC1",      #Consonant NA
                u"\u0A24":u"\xC2",      #Consonant
                u"\u0A25":u"\xC3",      #Consonant
                u"\u0A26":u"\xC4",      #Consonant
                u"\u0A27":u"\xC5",      #Consonant
                u"\u0A28":u"\xC6",      #Consonant NA
                u"\u0A29":u"\xC7",      #Consonant NNNA
                u"\u0A2A":u"\xC8",      #Consonant PA
                u"\u0A2B":u"\xC9",      #Consonant PHA
                u"\u0A2C":u"\xCA",      #Consonant BA
                u"\u0A2D":u"\xCB",      #Consonant BHA
                u"\u0A2E":u"\xCC",      #Consonant MA
                u"\u0A2F":u"\xCD",      #Consonant YA
                u"\u0A30":u"\xCF",      #Consonant RA
                u"\u0A31":u"\xD0",      #Consonant RRA
                u"\u0A32":u"\xD1",      #Consonant LA
                u"\u0A33":u"\xD2",      #Consonant LLA
                u"\u0A34":u"\xD3",      #Consonant LLLA
                u"\u0A35":u"\xD4",      #Consonant VA
                u"\u0A36":u"\xD5",      #Consonant SHA
                u"\u0A37":u"\xD6",      #Consonant SSA
                u"\u0A38":u"\xD7",      #Consonant SA
                u"\u0A39":u"\xD8",      #Consonant HA
                u"\u0A3A":u"",          #Consonant
                u"\u0A3B":u"",          #Consonant
                u"\u0A3C":u"\xE9",      #Consonant NUKTA
                u"\u0A3D":u"",          #Consonant AVAGRAHA
                u"\u0A3E":u"\xDA",      #Vowel Sign AA
                u"\u0A3F":u"\xDB",      #Vowel Sign I
                u"\u0A40":u"\xDC",      #Vowel Sign II
                u"\u0A41":u"\xDD",      #Vowel Sign U
                u"\u0A42":u"\xDE",      #Vowel 
                u"\u0A43":u"\xDF",      #Vowel
                u"\u0A44":u"",          #Vowel
                u"\u0A45":u"",          #Vowel
                u"\u0A46":u"\xE0",      #Vowel
                u"\u0A47":u"\xE1",      #Vowel
                u"\u0A48":u"\xE2",      #Vowel
                u"\u0A49":u"\xE7",      #Vowel
                u"\u0A4A":u"\xE4",      #Vowel
                u"\u0A4B":u"\xE5",      #Vowel
                u"\u0A4C":u"\xE6",      #Vowel
                u"\u0A4D":u"\xE8",      #Vowel Omission Sign Halant
                u"\u0A5C":u"\xE8",      #Vowel Omission Sign Halant
                u"\u0A64":u"\xEA",      #PURNA VIRAM has to be handled separately
                u"\u0A66":u"\xF1",      #Consonant
                u"\u0A67":u"\xF2",      #Consonant
                u"\u0A68":u"\xF3",      #Consonant
                u"\u0A69":u"\xF4",      #Consonant
                u"\u0A6A":u"\xF5",      #Consonant
                u"\u0A6B":u"\xF6",
                u"\u0A6C":u"\xF7",
                u"\u0A6D":u"\xF8",
                u"\u0A6E":u"\xF9",
                u"\u0A6F":u"\xFA",
                u"\u0A70":u"\xA1",      #Vowel-modifier GURMUKHI TIPPI
                u"\u0A71":u"\xFB",      #GURMUKHI ADDAK
                }
        self.unicode_norm_hashp_u2i = {
                u"\u0A59":u"\u0A16",
                u"\u0A5A":u"\u0A17",
                u"\u0A5B":u"\u0A1C",
                u"\u0A5E":u"\u0A2B",
                }
        self.hashk_u2i = {
                u"\u0C82":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0C83":u"\xA3",      #Vowel-modifier VISARG
                u"\u0C84":u"",
                u"\u0C85":u"\xA4",      #Vowel A
                u"\u0C86":u"\xA5",      #Vowel AA
                u"\u0C87":u"\xA6",      #Vowel I
                u"\u0C88":u"\xA7",      #Vowel II
                u"\u0C89":u"\xA8",      #Vowel U
                u"\u0C8A":u"\xA9",      #Vowel UU
                u"\u0C8B":u"\xAA",      #Vowel RI
                u"\u0C8C":u"",
                u"\u0C8D":u"\xAE",
                u"\u0C8E":u"\xAB",
                u"\u0C8F":u"\xAC",
                u"\u0C90":u"\xAD",
                u"\u0C91":u"\xB2",
                u"\u0C92":u"\xAF",
                u"\u0C93":u"\xB0",
                u"\u0C94":u"\xB1",
                u"\u0C95":u"\xB3",      #Consonant KA
                u"\u0C96":u"\xB4",      #Consonant
                u"\u0C97":u"\xB5",      #Consonant
                u"\u0C98":u"\xB6",      #Consonant
                u"\u0C99":u"\xB7",      #Consonant NGA
                u"\u0C9A":u"\xB8",      #Consonant
                u"\u0C9B":u"\xB9",      #Consonant
                u"\u0C9C":u"\xBA",      #Consonant
                u"\u0C9D":u"\xBB",      #Consonant
                u"\u0C9E":u"\xBC",      #Consonant JNA
                u"\u0C9F":u"\xBD",      #Consonant
                u"\u0CA0":u"\xBE",      #Consonant
                u"\u0CA1":u"\xBF",      #Consonant
                u"\u0CA2":u"\xC0",      #Consonant
                u"\u0CA3":u"\xC1",      #Consonant NA
                u"\u0CA4":u"\xC2",      #Consonant
                u"\u0CA5":u"\xC3",      #Consonant
                u"\u0CA6":u"\xC4",      #Consonant
                u"\u0CA7":u"\xC5",      #Consonant
                u"\u0CA8":u"\xC6",      #Consonant NA
                u"\u0CA9":u"\xC7",      #Consonant NNNA
                u"\u0CAA":u"\xC8",      #Consonant PA
                u"\u0CAB":u"\xC9",      #Consonant PHA
                u"\u0CAC":u"\xCA",      #Consonant BA
                u"\u0CAD":u"\xCB",      #Consonant BHA
                u"\u0CAE":u"\xCC",      #Consonant MA
                u"\u0CAF":u"\xCD",      #Consonant YA
                u"\u0CB0":u"\xCF",      #Consonant RA
                u"\u0CB1":u"\xD0",      #Consonant RRA
                u"\u0CB2":u"\xD1",      #Consonant LA
                u"\u0CB3":u"\xD2",      #Consonant LLA
                u"\u0CB4":u"\xD3",      #Consonant LLLA
                u"\u0CB5":u"\xD4",      #Consonant VA
                u"\u0CB6":u"\xD5",      #Consonant SHA
                u"\u0CB7":u"\xD6",      #Consonant SSA
                u"\u0CB8":u"\xD7",      #Consonant SA
                u"\u0CB9":u"\xD8",      #Consonant HA
                u"\u0CBC":u"\xE9",      #Consonant NUKTA
                u"\u0CBD":u"",          #Consonant AVAGRAHA
                u"\u0CBE":u"\xDA",      #Vowel Sign AA
                u"\u0CBF":u"\xDB",      #Vowel Sign I
                u"\u0CC0":u"\xDC",      #Vowel Sign II
                u"\u0CC1":u"\xDD",      #Vowel Sign U
                u"\u0CC2":u"\xDE",      #Vowel 
                u"\u0CC3":u"\xDF",      #Vowel
                u"\u0CC4":u"",          #Vowel
                u"\u0CC5":u"",          #Vowel
                u"\u0CC6":u"\xE0",      #Vowel
                u"\u0CC7":u"\xE1",      #Vowel
                u"\u0CC8":u"\xE2",      #Vowel
                u"\u0CC9":u"\xE7",      #Vowel
                u"\u0CCA":u"\xE4",      #Vowel
                u"\u0CCB":u"\xE5",      #Vowel
                u"\u0CCC":u"\xE6",      #Vowel
                u"\u0CCD":u"\xE8",      #Consonant
                u"\u0CD0":u"",          #Consonant
                u"\u0CE4":u"\xEA",      #Consonant
                u"\u0CE6":u"\xF1",      #Consonant
                u"\u0CE7":u"\xF2",      #Consonant
                u"\u0CE8":u"\xF3",      #Consonant
                u"\u0CE9":u"\xF4",      #Consonant
                u"\u0CEA":u"\xF5",      #Consonant
                u"\u0CEB":u"\xF6",
                u"\u0CEC":u"\xF7",
                u"\u0CED":u"\xF8",
                u"\u0CEE":u"\xF9",
                u"\u0CEF":u"\xFA",
                }
        self.unicode_norm_hashk_u2i = {
                u"\u0958":u"\u0915",
                u"\u0959":u"\u0916",
                u"\u095A":u"\u0917",
                u"\u095B":u"\u091C",
                u"\u095C":u"\u0921",
                u"\u095D":u"\u0922",
                u"\u095E":u"\u092B",
                u"\u095F":u"\u092F",
                }
        self.hashm_u2i = { 
                u"\u0D02":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0D03":u"\xA3",      #Vowel-modifier VISARG
                u"\u0D04":u"",
                u"\u0D05":u"\xA4",      #Vowel A
                u"\u0D06":u"\xA5",      #Vowel AA
                u"\u0D07":u"\xA6",      #Vowel I
                u"\u0D08":u"\xA7",      #Vowel II
                u"\u0D09":u"\xA8",      #Vowel U
                u"\u0D0A":u"\xA9",      #Vowel UU
                u"\u0D0B":u"\xAA",      #Vowel RI
                u"\u0D0C":u"",
                u"\u0D0D":u"",
                u"\u0D0E":u"\xAB",
                u"\u0D0F":u"\xAC",
                u"\u0D10":u"\xAD",
                u"\u0D11":u"\xB2",
                u"\u0D12":u"\xAF",
                u"\u0D13":u"\xB0",
                u"\u0D14":u"\xB1",
                u"\u0D15":u"\xB3",      #Consonant KA
                u"\u0D16":u"\xB4",      #Consonant
                u"\u0D17":u"\xB5",      #Consonant
                u"\u0D18":u"\xB6",      #Consonant
                u"\u0D19":u"\xB7",      #Consonant NGA
                u"\u0D1A":u"\xB8",      #Consonant
                u"\u0D1B":u"\xB9",      #Consonant
                u"\u0D1C":u"\xBA",      #Consonant
                u"\u0D1D":u"\xBB",      #Consonant
                u"\u0D1E":u"\xBC",      #Consonant JNA
                u"\u0D1F":u"\xBD",      #Consonant
                u"\u0D20":u"\xBE",      #Consonant
                u"\u0D21":u"\xBF",      #Consonant
                u"\u0D22":u"\xC0",      #Consonant
                u"\u0D23":u"\xC1",      #Consonant NNA
                u"\u0D24":u"\xC2",      #Consonant
                u"\u0D25":u"\xC3",      #Consonant
                u"\u0D26":u"\xC4",      #Consonant
                u"\u0D27":u"\xC5",      #Consonant
                u"\u0D28":u"\xC6",      #Consonant NA
                u"\u0D29":u"\xC7",      #Consonant NNNA
                u"\u0D2A":u"\xC8",      #Consonant PA
                u"\u0D2B":u"\xC9",      #Consonant PHA
                u"\u0D2C":u"\xCA",      #Consonant BA
                u"\u0D2D":u"\xCB",      #Consonant BHA
                u"\u0D2E":u"\xCC",      #Consonant MA
                u"\u0D2F":u"\xCD",      #Consonant YA
                u"\u0D30":u"\xCF",      #Consonant RA
                u"\u0D31":u"\xD0",      #Consonant RRA
                u"\u0D32":u"\xD1",      #Consonant LA
                u"\u0D33":u"\xD2",      #Consonant LLA
                u"\u0D34":u"\xD3",      #Consonant LLLA
                u"\u0D35":u"\xD4",      #Consonant VA
                u"\u0D36":u"\xD5",      #Consonant SHA
                u"\u0D37":u"\xD6",      #Consonant SSA
                u"\u0D38":u"\xD7",      #Consonant SA
                u"\u0D39":u"\xD8",      #Consonant HA
                u"\u0D3D":u"",          #Consonant AVAGRAHA
                u"\u0D3E":u"\xDA",      #Vowel Sign AA
                u"\u0D3F":u"\xDB",      #Vowel Sign I
                u"\u0D40":u"\xDC",      #Vowel Sign II
                u"\u0D41":u"\xDD",      #Vowel Sign U
                u"\u0D42":u"\xDE",      #Vowel
                u"\u0D43":u"\xDF",      #Vowel
                u"\u0D44":u"",          #Vowel
                u"\u0D45":u"",          #Vowel
                u"\u0D46":u"\xE0",      #Vowel
                u"\u0D47":u"\xE1",      #Vowel
                u"\u0D48":u"\xE2",      #Vowel
                u"\u0D4A":u"\xE4",      #Vowel
                u"\u0D4B":u"\xE5",      #Vowel
                u"\u0D4C":u"\xE6",      #Vowel
                u"\u0D4D":u"\xE8",      #Consonant
                u"\u0D64":u"\xEA",      #Consonant
                u"\u0D66":u"\xF1",      #Consonant
                u"\u0D67":u"\xF2",      #Consonant
                u"\u0D68":u"\xF3",      #Consonant
                u"\u0D69":u"\xF4",      #Consonant
                u"\u0D6A":u"\xF5",      #Consonant
                u"\u0D6B":u"\xF6",
                u"\u0D6C":u"\xF7",
                u"\u0D6D":u"\xF8",
                u"\u0D6E":u"\xF9",
                u"\u0D6F":u"\xFA",
                }
        self.hashb_u2i = { 
                u"\u0981":u"\xA1",      #vowel-modifier CHANDRABINDU 
                u"\u0982":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0983":u"\xA3",      #Vowel-modifier VISARG
                u"\u0985":u"\xA4",      #Vowel A
                u"\u0986":u"\xA5",      #Vowel AA
                u"\u0987":u"\xA6",      #Vowel I
                u"\u0988":u"\xA7",      #Vowel II
                u"\u0989":u"\xA8",      #Vowel U
                u"\u098A":u"\xA9",      #Vowel UU
                u"\u098B":u"\xAA",      #Vowel RI
                u"\u098C":u"",
                u"\u098D":u"",
                u"\u098E":u"",
                u"\u098F":u"\xAB",
                u"\u0990":u"\xAD",
                u"\u0991":u"",
                u"\u0992":u"",
                u"\u0993":u"\xAF",
                u"\u0994":u"\xB1",
                u"\u0995":u"\xB3",      #Consonant KA
                u"\u0996":u"\xB4",      #Consonant
                u"\u0997":u"\xB5",      #Consonant
                u"\u0998":u"\xB6",      #Consonant
                u"\u0999":u"\xB7",      #Consonant NGA
                u"\u099A":u"\xB8",      #Consonant
                u"\u099B":u"\xB9",      #Consonant
                u"\u099C":u"\xBA",      #Consonant
                u"\u099D":u"\xBB",      #Consonant
                u"\u099E":u"\xBC",      #Consonant JNA
                u"\u099F":u"\xBD",      #Consonant
                u"\u09A0":u"\xBE",      #Consonant
                u"\u09A1":u"\xBF",      #Consonant
                u"\u09A2":u"\xC0",      #Consonant
                u"\u09A3":u"\xC1",      #Consonant NA
                u"\u09A4":u"\xC2",      #Consonant
                u"\u09A5":u"\xC3",      #Consonant
                u"\u09A6":u"\xC4",      #Consonant
                u"\u09A7":u"\xC5",      #Consonant
                u"\u09A8":u"\xC6",      #Consonant NA
                u"\u09A9":u"",      
                u"\u09AA":u"\xC8",      #Consonant PA
                u"\u09AB":u"\xC9",      #Consonant PHA
                u"\u09AC":u"\xCA",      #Consonant BA
                u"\u09AD":u"\xCB",      #Consonant BHA
                u"\u09AE":u"\xCC",      #Consonant MA
                u"\u09AF":u"\xCD",      #Consonant YA
                u"\u09B0":u"\xCF",      #Consonant RA
                u"\u09B1":u"",       
                u"\u09B2":u"\xD1",      #Consonant LA
                u"\u09B3":u"",       
                u"\u09B4":u"",     
                u"\u09B5":u"",           
                u"\u09B6":u"\xD5",      #Consonant SHA
                u"\u09B7":u"\xD6",      #Consonant SSA
                u"\u09B8":u"\xD7",      #Consonant SA
                u"\u09B9":u"\xD8",      #Consonant HA
                u"\u09BC":u"\xE9",      #Consonant NUKTA
                u"\u09BD":u"",          #Consonant AVAGRAHA
                u"\u09BE":u"\xDA",      #Vowel Sign AA
                u"\u09BF":u"\xDB",      #Vowel Sign I
                u"\u09C0":u"\xDC",      #Vowel Sign II
                u"\u09C1":u"\xDD",      #Vowel Sign U
                u"\u09C2":u"\xDE",      #Vowel
                u"\u09C3":u"\xDF",      #Vowel
                u"\u09C4":u"",          #Vowel
                u"\u09C5":u"",          #Vowel
                u"\u09C6":u"",          #Vowel
                u"\u09C7":u"\xE0",      #Vowel
                u"\u09C8":u"\xE2",      #Vowel
                u"\u09CB":u"\xE4",      #Vowel
                u"\u09CC":u"\xE6",      #Vowel
                u"\u09CD":u"\xE8",      #Consonant
                u"\u09CE":u"",          #Consonant
                u"\u09E4":u"\xEA",      #Consonant
                u"\u0964":u"\xEA",      #Consonant
                u"\u09E6":u"\xF1",      #Consonant
                u"\u09E7":u"\xF2",      #Consonant
                u"\u09E8":u"\xF3",      #Consonant
                u"\u09E9":u"\xF4",      #Consonant
                u"\u09EA":u"\xF5",      #Consonant
                u"\u09EB":u"\xF6",
                u"\u09EC":u"\xF7",
                u"\u09ED":u"\xF8",
                u"\u09EE":u"\xF9",
                u"\u09EF":u"\xFA",
                }
        self.hashta_u2i = { 
                u"\u0B82":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0B83":u"\xA3",      #Vowel-modifier VISARG
                u"\u0B85":u"\xA4",      #Vowel A
                u"\u0B86":u"\xA5",      #Vowel AA
                u"\u0B87":u"\xA6",      #Vowel I
                u"\u0B88":u"\xA7",      #Vowel II
                u"\u0B89":u"\xA8",      #Vowel U
                u"\u0B8A":u"\xA9",      #Vowel UU
                u"\u0B8B":u"",          #Vowel RI
                u"\u0B8C":u"",
                u"\u0B8D":u"",
                u"\u0B8E":u"\xAB",
                u"\u0B8F":u"\xAC",
                u"\u0B90":u"\xAD",
                u"\u0B91":u"",
                u"\u0B92":u"\xAF",      #check here
                u"\u0B93":u"\xB0",
                u"\u0B94":u"\xB1",
                u"\u0B95":u"\xB3",      #Consonant KA
                u"\u0B96":u"",          #Consonant
                u"\u0B97":u"",          #Consonant
                u"\u0B98":u"",          #Consonant
                u"\u0B99":u"\xB7",      #Consonant NGA
                u"\u0B9A":u"\xB8",      #Consonant
                u"\u0B9B":u"",          #Consonant
                u"\u0B9C":u"\xBA",      #Consonant
                u"\u0B9D":u"",          #Consonant
                u"\u0B9E":u"\xBC",      #Consonant JNA
                u"\u0B9F":u"\xBD",      #Consonant
                u"\u0BA0":u"",          #Consonant
                u"\u0BA1":u"",          #Consonant
                u"\u0BA2":u"",          #Consonant
                u"\u0BA3":u"\xC1",      #Consonant NA
                u"\u0BA4":u"\xC2",      #Consonant
                u"\u0BA5":u"",          #Consonant
                u"\u0BA6":u"",          #Consonant
                u"\u0BA7":u"",          #Consonant
                u"\u0BA8":u"\xC6",      #Consonant NA
                u"\u0BA9":u"\xC7",      #Consonant NNNA
                u"\u0BAA":u"\xC8",      #Consonant PA
                u"\u0BAB":u"",          #Consonant PHA
                u"\u0BAC":u"",          #Consonant BA
                u"\u0BAD":u"",          #Consonant BHA
                u"\u0BAE":u"\xCC",      #Consonant MA
                u"\u0BAF":u"\xCD",      #Consonant YA
                u"\u0BB0":u"\xCF",      #Consonant RA
                u"\u0BB1":u"\xD0",      #Consonant RRA
                u"\u0BB2":u"\xD1",      #Consonant LA
                u"\u0BB3":u"\xD2",      #Consonant LLA
                u"\u0BB4":u"\xD3",      #Consonant LLLA
                u"\u0BB5":u"\xD4",      #Consonant VA
                u"\u0BB6":u"\xD5",      #Consonant SHA
                u"\u0BB7":u"\xD6",      #Consonant SSA
                u"\u0BB8":u"\xD7",      #Consonant SA
                u"\u0BB9":u"\xD8",      #Consonant HA
                u"\u0BBE":u"\xDA",      #Vowel Sign AA
                u"\u0BBF":u"\xDB",      #Vowel Sign I
                u"\u0BC0":u"\xDC",      #Vowel Sign II
                u"\u0BC1":u"\xDD",      #Vowel Sign U
                u"\u0BC2":u"\xDE",      #Vowel
                u"\u0BC3":u"",          #Vowel
                u"\u0BC4":u"",          #Vowel
                u"\u0BC5":u"",          #Vowel
                u"\u0BC6":u"\xE0",      #Vowel
                u"\u0BC7":u"\xE1",      #Vowel
                u"\u0BC8":u"\xE2",      #Vowel
                u"\u0BCA":u"\xE4",      #Vowel ொ
                u"\u0BCB":u"\xE5",      #Vowel ோ
                u"\u0BCC":u"\xE6",      #Vowel ௌ
                u"\u0BCD":u"\xE8",      #Halant
                u"\u0BD0":u"",          #Consonant
                u"\u0BD1":u"",
                u"\u0BD2":u"",
                u"\u0BD3":u"",
                u"\u0BD4":u"",
                u"\u0BD5":u"",
                u"\u0BD6":u"",
                u"\u0BD7":u"",
                u"\u0BE4":u"\xEA",      #Consonant
                u"\u0BE5":u"\xEA", 
                u"\u0BE6":u"\xF1",      #Consonant
                u"\u0BE7":u"\xF2",      #Consonant
                u"\u0BE8":u"\xF3",      #Consonant
                u"\u0BE9":u"\xF4",      #Consonant
                u"\u0BEA":u"\xF5",      #Consonant
                u"\u0BEB":u"\xF6",
                u"\u0BEC":u"\xF7",
                u"\u0BED":u"\xF8",
                u"\u0BEE":u"\xF9",
                u"\u0BEF":u"\xFA",
                }
        self.hasho_u2i = {
                u"\u0B01":u"\xA1",      #Vowel-modifier CHANDRABINDU
                u"\u0B02":u"\xA2",      #Vowel-modifier ANUSWAR
                u"\u0B03":u"\xA3",      #Vowel-modifier VISARG
                u"\u0904":u"",
                u"\u0B05":u"\xA4",      #Vowel A
                u"\u0B06":u"\xA5",      #Vowel AA
                u"\u0B07":u"\xA6",      #Vowel I
                u"\u0B08":u"\xA7",      #Vowel II
                u"\u0B09":u"\xA8",      #Vowel U
                u"\u0B0A":u"\xA9",      #Vowel UU
                u"\u0B0B":u"\xAA",      #Vowel RI
                u"\u0B0C":u"",
                u"\u0B0D":u"",
                u"\u0B0E":u"",
                u"\u0B0F":u"\xAC",
                u"\u0910":u"\xAD",
                u"\u0B11":u"",
                u"\u0B12":u"",
                u"\u0B13":u"\xB0",
                u"\u0B14":u"\xB1",
                u"\u0B15":u"\xB3",      #Consonant KA
                u"\u0B16":u"\xB4",      #Consonant
                u"\u0B17":u"\xB5",      #Consonant
                u"\u0B18":u"\xB6",      #Consonant
                u"\u0B19":u"\xB7",      #Consonant NGA
                u"\u0B1A":u"\xB8",      #Consonant
                u"\u0B1B":u"\xB9",      #Consonant
                u"\u0B1C":u"\xBA",      #Consonant
                u"\u0B1D":u"\xBB",      #Consonant
                u"\u0B1E":u"\xBC",      #Consonant JNA
                u"\u0B1F":u"\xBD",      #Consonant
                u"\u0B20":u"\xBE",      #Consonant
                u"\u0B21":u"\xBF",      #Consonant
                u"\u0B22":u"\xC0",      #Consonant
                u"\u0B23":u"\xC1",      #Consonant NA
                u"\u0B24":u"\xC2",      #Consonant
                u"\u0B25":u"\xC3",      #Consonant
                u"\u0B26":u"\xC4",      #Consonant
                u"\u0B27":u"\xC5",      #Consonant
                u"\u0B28":u"\xC6",      #Consonant NA
                u"\u0929":u"",          #Consonant NNNA
                u"\u0B2A":u"\xC8",      #Consonant PA
                u"\u0B2B":u"\xC9",      #Consonant PHA
                u"\u0B2C":u"\xCA",      #Consonant BA
                u"\u0B2D":u"\xCB",      #Consonant BHA
                u"\u0B2E":u"\xCC",      #Consonant MA
                u"\u0B2F":u"\xCD",      #Consonant YA
                u"\u0B30":u"\xCF",      #Consonant RA
                u"\u0931":u"",          #Consonant RRA
                u"\u0B32":u"\xD1",      #Consonant LA
                u"\u0B33":u"\xD2",      #Consonant LLA
                u"\u0934":u"",          #Consonant LLLA
                u"\u0935":u"\xD4",      #Consonant VA
                u"\u0B36":u"\xD5",      #Consonant SHA
                u"\u0B37":u"\xD6",      #Consonant SSA
                u"\u0B38":u"\xD7",      #Consonant SA
                u"\u0B39":u"\xD8",      #Consonant HA
                u"\u0B3A":u"",          #Consonant
                u"\u0B3B":u"",          #Consonant
                u"\u0B3C":u"\xE9",      #Consonant NUKTA
                u"\u0B3D":u"",          #Consonant AVAGRAHA
                u"\u0B3E":u"\xDA",      #Vowel Sign AA
                u"\u0B3F":u"\xDB",      #Vowel Sign I
                u"\u0B40":u"\xDC",      #Vowel Sign II
                u"\u0B41":u"\xDD",      #Vowel Sign U
                u"\u0B42":u"\xDE",      #Vowel 
                u"\u0B43":u"\xDF",      #Vowel
                u"\u0B44":u"",          #Vowel
                u"\u0B45":u"",          #Vowel
                u"\u0B46":u"",          #Vowel
                u"\u0B47":u"\xE1",      #Vowel
                u"\u0B48":u"\xE2",      #Vowel
                u"\u0B49":u"",          #Vowel
                u"\u0B4A":u"",          #Vowel
                u"\u0B4B":u"\xE5",      #Vowel O
                u"\u0B4C":u"\xE6",      #Vowel OU
                u"\u0B4D":u"\xE8",      # Halant
                u"\u0B50":u"",          #Consonant
                u"\u0964":u"\xEA",      #Full stop use Devanagri
                u"\u0B66":u"\xF1",      # Digit 0
                u"\u0B67":u"\xF2",      # Digit 1
                u"\u0B68":u"\xF3",      # Digit 2
                u"\u0B69":u"\xF4",      # Digit 3
                u"\u0B6A":u"\xF5",      # Digit 4
                u"\u0B6B":u"\xF6",      # Digit 5
                u"\u0B6C":u"\xF7",      # Digit 6
                u"\u0B6D":u"\xF8",      # Digit 7
                u"\u0B6E":u"\xF9",      # Digit 8
                u"\u0B6F":u"\xFA",      # Digit 9
                }
            #unicode_norm_hasho_u2i = {
            #   u"\x0958":u"\u0915",
            #   u"\x0959":u"\u0916",
            #   u"\x095A":u"\u0917",
            #   u"\x095B":u"\u091C",
            #   u"\x095C":u"\u0921",
            #   u"\x095D":u"\u0922",
            #   u"\x095E":u"\u092B",
            #   u"\x095F":u"\u092F",
            #   }

        # compile regexes
        self.c = re.compile(u"([\xB3-\xD8])")
        self.v = re.compile(u"([\xA5-\xB2])")
        self.dig = re.compile(u"([\xF1-\xFA])")
        self.ch = re.compile(u"([\xB3-\xD8])\xE8")
        self.cn = re.compile(u"([\xB3-\xD8])\xE9")
        self.amd = re.compile(u"[\xA4]([\xA1-\xA3])")
        self.cm = re.compile(u"([\xB3-\xD8])([\xDA-\xE7])")
        self.vmd = re.compile(u"([\xA5-\xB2])([\xA1-\xA3])")
        self.cnh = re.compile(u"([\xB3-\xD8])\xE9\xE8")
        self.cmd = re.compile(u"([\xB3-\xD8])([\xA1-\xA3])")
        self.cnm = re.compile(u"([\xB3-\xD8])\xE9([\xDA-\xE7])")
        self.cnmd = re.compile(u"([\xB3-\xD8])\xE9([\xA1-\xA3])")
        self.cmmd = re.compile(u"([\xB3-\xD8])([\xDA-\xE7])([\xA1-\xA3])")
        self.cnmmd = re.compile(u"([\xB3-\xD8])\xE9([\xDA-\xE7])([\xA1-\xA3])")

        self.u2i_hn = re.compile(u"([\u0958-\u095F])")
        self.u2i_h = re.compile(u"([\u0900-\u097F])")
        self.u2i_t = re.compile(u"([\u0C01-\u0C6F])")
        self.u2i_k = re.compile(u"([\u0C80-\u0CFF])")
        self.u2i_m = re.compile(u"([\u0D00-\u0D6F])")
        self.u2i_b = re.compile(u"([\u0980-\u09EF\u0964])")
        self.u2i_o = re.compile(u"([\u0B00-\u0B7F\u0964])")
        self.u2i_p = re.compile(u"([\u0A01-\u0A75])")
        self.u2i_ta = re.compile(u"([\u0B82-\u0BEF])")
        self.u2i_kn = re.compile(u"([\u0958-\u095F])")
        self.u2i_pn = re.compile(u"([\u0A59-\u0A5B\u0A5E])")

    def normalize(self,text):
        """
        Performs some common normalization, which includes: 
        - Byte order mark, word joiner, etc. removal 
        - ZERO_WIDTH_NON_JOINER and ZERO_WIDTH_JOINER removal 
        """
        text = text.replace(self.BYTE_ORDER_MARK,'')
        text = text.replace(self.BYTE_ORDER_MARK_2,'')
        text = text.replace(self.WORD_JOINER,'')
        text = text.replace(self.SOFT_HYPHEN,'')

        text = text.replace(self.ZERO_WIDTH_NON_JOINER, '')
        text = text.replace(self.ZERO_WIDTH_JOINER,'')

        return text

    def wx2iscii(self, my_string):
        """Convert WX to ISCII"""
        Z, _u_ = u'Z' in my_string, u'_' in my_string
        MHz = u'M' in my_string or u'H' in my_string or u'z' in my_string 
        eV, EY, oV, OY = u'eV' in my_string, u'EY' in my_string, u'oV' in my_string, u'OY' in my_string
        q, l, n, rY, a = u'q' in my_string, u'l' in my_string, u'n' in my_string, u'rY' in my_string, u'a' in my_string
        if _u_:
            my_string = re.sub(u'k_ReV([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"eV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'k_ReV', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"eV"])
            my_string = re.sub(u'k_REY([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'k_REY', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"EY"])
            my_string = re.sub(u'k_RoV([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'k_RoV', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"oV"])
            my_string = re.sub(u'k_ROY([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'k_ROY', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashm_w2i[u"OY"])
            my_string = re.sub(u'k_R([AiIuUoO])([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"R"]+self.hashm_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'k_R([AiIuUoO])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                                self.hashm_w2i[m.group(1)], my_string)
            my_string = re.sub(u'k_Ra([MHz])', lambda m: self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'k_Ra', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"])
            my_string = my_string.replace(u'k_R', self.hashc_w2i[u"k"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"R"]+
                        self.hashc_w2i[u"_"])
            my_string = re.sub(u'w_reV([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"eV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'w_reV', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"eV"])
            my_string = re.sub(u'w_rEY([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'w_rEY', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"EY"])
            my_string = re.sub(u'w_roV([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'w_roV', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"oV"])
            my_string = re.sub(u'w_rOY([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'w_rOY', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"OY"])
            my_string = re.sub(u'w_r([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"r"]+self.hashm_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'w_r([AiIuUeEoO])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"r"]+self.hashm_w2i[m.group(1)], my_string)
            my_string = re.sub(u'w_ra([MHz])', lambda m: self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'w_ra', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"])
            my_string = my_string.replace(u'w_r', self.hashc_w2i[u"w"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashc_w2i[u"_"])
            my_string = re.sub(u'j_FeV([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"eV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'j_FeV', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"eV"])
            my_string = re.sub(u'j_FEY([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'j_FEY', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"EY"])
            my_string = re.sub(u'j_FoV([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'j_FoV', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"oV"])
            my_string = re.sub(u'j_FOY([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'j_FOY', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashm_w2i[u"OY"])
            my_string = re.sub(u'j_F([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"F"]+self.hashm_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'j_F([AiIuUeEoO])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"F"]+self.hashm_w2i[m.group(1)], my_string)
            my_string = re.sub(u'j_Fa([MHz])', lambda m: self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'j_Fa', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"])
            my_string = my_string.replace(u'j_F', self.hashc_w2i[u"j"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"F"]+
                        self.hashc_w2i[u"_"])
            my_string = re.sub(u'S_reV([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"eV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'S_reV', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"eV"])
            my_string = re.sub(u'S_rEY([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'S_rEY', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"EY"])
            my_string = re.sub(u'S_roV([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'S_roV', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"oV"])
            my_string = re.sub(u'S_rOY([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'S_rOY', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[u"OY"])
            my_string = re.sub(u'S_r([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+
                        self.hashc_w2i[u"r"]+self.hashm_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'S_r([AiIuUeEoO])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashm_w2i[m.group(1)], my_string)
            my_string = re.sub(u'S_ra([MHz])', lambda m: self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'S_ra', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"])
            my_string = my_string.replace(u'S_r', self.hashc_w2i[u"S"]+self.hashc_w2i[u"_"]+self.hashc_w2i[u"r"]+
                        self.hashc_w2i[u"r"])
    
        if eV:
            if MHz:
                my_string = self.ccceVmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aeV"]+self.hashmd_w2i[m.group(4)], my_string)
            my_string = self.ccceV.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aeV"], my_string)
        if EY:
            if MHz:
                my_string = self.cccEYmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aEY"]+self.hashmd_w2i[m.group(4)], my_string)
            my_string = self.cccEY.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aEY"], my_string)
        if oV:
            if MHz:
                my_string = self.cccoVmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aoV"]+self.hashmd_w2i[m.group(4)], my_string)
            my_string = self.cccoV.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aoV"], my_string)
        if OY:
            if MHz:
                my_string = self.cccOYmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aOY"]+self.hashmd_w2i[m.group(4)], my_string)
            my_string = self.cccOY.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[u"aOY"], my_string)

        if MHz:
            my_string = self.cccvmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[m.group(4)]+self.hashmd_w2i[m.group(5)], my_string)
        my_string = self.cccv.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                    self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashm_w2i[m.group(4)], my_string)
        if a:
            if MHz:
                my_string = self.cccamd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashmd_w2i[m.group(4)], my_string)
            my_string = self.ccca.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)], my_string)
        #NOTE consonant+consonant+consonant moved from here
        if eV:
            if MHz:
                my_string = self.cceVmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashm_w2i[u"eV"]+self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.cceV.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[u"eV"], my_string)
        if EY:
            if MHz:
                my_string = self.ccEYmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+
                            self.hashc_w2i[m.group(2)]+self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.ccEY.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[u"EY"], my_string)
        if oV:
            if MHz:
                my_string = self.ccoVmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.ccoV.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[u"oV"], my_string)
        if OY:
            if MHz:
                my_string = self.ccOYmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.ccOY.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[u"OY"], my_string)

        if MHz:
            my_string = self.ccvmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[m.group(3)]+self.hashmd_w2i[m.group(4)], my_string)
        my_string = self.ccv.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                        self.hashm_w2i[m.group(3)], my_string)
        if a:
            if MHz:
                my_string = self.ccamd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                            self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.cca.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)], my_string)
        if eV:
            if MHz:
                my_string = self.ceVmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"]
                            +self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.ceV.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"], my_string)
        if EY:
            if MHz:
                my_string = self.cEYmd.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.cEY.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"], my_string)
        if oV:
            if MHz:
                my_string = self.coVmd.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.coV.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"], my_string)
        if OY:
            if Z:   #NOTE Case ZOY added
                my_string = self.cZOY.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+self.hashm_w2i[u"OY"], my_string)
            if MHz:
                my_string = self.cOYmd.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.cOY.sub( lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"], my_string)

        if Z:
            if MHz:
                my_string = self.cZvmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+
                            self.hashm_w2i[m.group(2)]+self.hashmd_w2i[m.group(3)], my_string)
            my_string = self.cZv.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+self.hashm_w2i[m.group(2)], my_string)
            if MHz:
                my_string = self.cZamd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+self.hashmd_w2i[m.group(2)], my_string)
                my_string = self.cZmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.cZa.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"], my_string)
            #NOTE consonant+YZa case added
            my_string = self.cYZa.sub(lambda m: self.hashc_w2i[m.group(1)+u"Y"]+self.hashc_w2i[u"Z"], my_string)
            my_string = self.cZ.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"Z"]+self.hashc_w2i[u"_"], my_string)
        #NOTE consonant+consonant+consonant replaced
        my_string = self.ccc.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(2)]+
                    self.hashc_w2i[u"_"]+self.hashc_w2i[m.group(3)]+self.hashc_w2i[u"_"], my_string)
        if q:
            if MHz:
                my_string = self.cqmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"q"]+self.hashmd_w2i[m.group(2)], my_string)
                #NOTE q+[MHz] case added
                my_string = self.qmd.sub(lambda m: self.hashv_w2i[u"q"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = self.cq.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"q"], my_string)
            #Added for the case Vowel(U090B)+Modifier
            if MHz:
                my_string = self.aqmd.sub(lambda m: self.hashv_w2i[u"aq"]+self.hashmd_w2i[m.group(1)], my_string)
            #NOTE q, aq removed from here
        #Added for the case lYYa,lYY[AiIuUeEoO],lYY[MHz]
        if l:
            if MHz:
                my_string = re.sub(u'(lYY)eV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lYY)eV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"], my_string)
            if MHz:
                my_string = re.sub(u'(lYY)EY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lYY)EY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"], my_string)
            if MHz:
                my_string = re.sub(u'(lYY)oV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lYY)oV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"], my_string)
            if MHz:
                my_string = re.sub(u'(lYY)OY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lYY)OY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"], my_string)
            if MHz:
                my_string = re.sub(u'(lYY)([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+
                            self.hashmd_w2i[m.group(3)], my_string)
            my_string = re.sub(u'(lYY)([AiIuUeEoO])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)], my_string)
            if MHz:
                my_string = re.sub(u'(lYY)a([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lYY)a', lambda m: self.hashc_w2i[m.group(1)], my_string)
            my_string = re.sub(u'(lYY)', lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"], my_string)
            #Added for the case lYa,lY[AiIuUeEoO],lY[MHz]
            if MHz:
                my_string = re.sub(u'(lY)eV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lY)eV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"], my_string)
            if MHz:
                my_string = re.sub(u'(lY)EY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lY)EY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"], my_string)
            if MHz:
                my_string = re.sub(u'(lY)oV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lY)oV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"], my_string)
            if MHz:
                my_string = re.sub(u'(lY)OY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lY)OY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"], my_string)
            if MHz:
                my_string = re.sub(u'(lY)([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+
                            self.hashmd_w2i[m.group(3)], my_string)
            my_string = re.sub(u'(lY)([AiIuUeEoO])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)], my_string)
            if MHz:
                my_string = re.sub(u'(lY)a([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(lY)a', lambda m: self.hashc_w2i[m.group(1)], my_string)
            my_string = re.sub(u'(lY)', lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"], my_string)
        #Added for tamil -by Rashid
        if n:
            if MHz:
                my_string = re.sub(u'(nY)eV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(nY)eV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"], my_string)
            if MHz:
                my_string = re.sub(u'(nY)EY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(nY)EY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"], my_string)
            if MHz:
                my_string = re.sub(u'(nY)oV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(nY)oV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"], my_string)
            if MHz:
                my_string = re.sub(u'(nY)OY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(nY)OY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"], my_string)
            if MHz:
                my_string = re.sub(u'(nY)([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+
                            self.hashmd_w2i[m.group(3)], my_string)
            my_string = re.sub(u'(nY)([AiIuUeEoO])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)], my_string)
            if MHz:
                my_string = re.sub(u'(nY)a([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(nY)a', lambda m: self.hashc_w2i[m.group(1)], my_string)
            my_string = re.sub(u'(nY)', lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"], my_string)
        #Added for tamil -by Rashid
        if rY:
            if MHz:
                my_string = re.sub(u'(rY)eV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(rY)eV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"eV"], my_string)
            if MHz:
                my_string = re.sub(u'(rY)EY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(rY)EY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"EY"], my_string)
            if MHz:
                my_string = re.sub(u'(rY)oV([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(rY)oV', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"oV"], my_string)
            if MHz:
                my_string = re.sub(u'(rY)OY([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"]+
                            self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(rY)OY', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[u"OY"], my_string)
            if MHz:
                my_string = re.sub(u'(rY)([AiIuUeEoO])([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+
                            self.hashmd_w2i[m.group(3)], my_string)
            my_string = re.sub(u'(rY)([AiIuUeEoO])', lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)], my_string)
            if MHz:
                my_string = re.sub(u'(rY)a([MHz])', lambda m: self.hashc_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = re.sub(u'(rY)a', lambda m: self.hashc_w2i[m.group(1)], my_string)
            my_string = re.sub(u'(rY)', lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"], my_string)

        if MHz:
            my_string = self.cvmd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+
                        self.hashmd_w2i[m.group(3)], my_string)
        #my_string = self.cvm.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+self.hashm_w2i[m.group(3)], my_string)
        #if Z:
        #    my_string = self.cZvm.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)]+self.hashm_w2i[m.group(3)], my_string)
        my_string = self.cv.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashm_w2i[m.group(2)], my_string)
        if a:
            if MHz:
                my_string = self.camd.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
            my_string = self.ca.sub(lambda m: self.hashc_w2i[m.group(1)], my_string)
        
        if q:
            #NOTE q, aq replaced 
            my_string = my_string.replace(u'aq', self.hashv_w2i[u"aq"])
            my_string = my_string.replace(u'q', self.hashv_w2i[u"aq"])

        my_string = self.c.sub(lambda m: self.hashc_w2i[m.group(1)]+self.hashc_w2i[u"_"], my_string)
        #Added for the case of U0946
        if eV:
            if MHz:
                my_string = re.sub(u'aeV([MHz])', lambda m: self.hashv_w2i[u"aeV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'aeV', self.hashv_w2i[u"aeV"])
            if MHz:
                my_string = re.sub(u'eV([MHz])', lambda m: self.hashv_w2i[u"eV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'eV', self.hashv_w2i[u"eV"])
        #Added for the case of U0945
        if EY:
            if MHz:
                my_string = re.sub(u'aEY([MHz])', lambda m: self.hashv_w2i[u"aEY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'aEY', self.hashv_w2i[u"aEY"])
            if MHz:
                my_string = re.sub(u'EY([MHz])', lambda m: self.hashv_w2i[u"EY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'EY', self.hashv_w2i[u"EY"])
        #Added for the case of U094A
        if oV:
            if MHz:
                my_string = re.sub(u'aoV([MHz])', lambda m: self.hashv_w2i[u"aoV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'aoV', self.hashv_w2i[u"aoV"])
            if MHz:
                my_string = re.sub(u'oV([MHz])', lambda m: self.hashv_w2i[u"oV"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'oV', self.hashv_w2i[u"oV"])
        #Added for the case of U0949
        if OY:
            if MHz:
                my_string = re.sub(u'aOY([MHz])', lambda m: self.hashv_w2i[u"aOY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'aOY', self.hashv_w2i[u"aOY"])
            if MHz:
                my_string = re.sub(u'OY([MHz])', lambda m: self.hashv_w2i[u"OY"]+self.hashmd_w2i[m.group(1)], my_string)
            my_string = my_string.replace(u'OY', self.hashv_w2i[u"OY"])
    
        if a:
            #NOTE zero-width non-word boundary added on the left of string
            #my_string = my_string.replace(u'aA', self.hashv_w2i[u"aA"])
            my_string = re.sub(u'\BaA', self.hashv_w2i[u"aA"], my_string)
            #my_string = my_string.replace(u'ai', self.hashv_w2i[u"ai"])
            my_string = re.sub(u'\Bai', self.hashv_w2i[u"ai"], my_string)
            #my_string = my_string.replace(u'aI', self.hashv_w2i[u"aI"])
            my_string = re.sub(u'\BaI', self.hashv_w2i[u"aI"], my_string)
            #my_string = my_string.replace(u'au', self.hashv_w2i[u"au"])
            my_string = re.sub(u'\Bau', self.hashv_w2i[u"au"], my_string)
            #my_string = my_string.replace(u'aU', self.hashv_w2i[u"aU"])
            my_string = re.sub(u'\BaU', self.hashv_w2i[u"aU"], my_string)
            #my_string = my_string.replace(u'ae', self.hashv_w2i[u"ae"])
            my_string = re.sub(u'\Bae', self.hashv_w2i[u"ae"], my_string)
            #my_string = my_string.replace(u'aE', self.hashv_w2i[u"aE"])
            my_string = re.sub(u'\BaE', self.hashv_w2i[u"aE"], my_string)
            #my_string = my_string.replace(u'ao', self.hashv_w2i[u"ao"])
            my_string = re.sub(u'\Bao', self.hashv_w2i[u"ao"], my_string)
            #my_string = my_string.replace(u'aO', self.hashv_w2i[u"aO"])
            my_string = re.sub(u'\BaO', self.hashv_w2i[u"aO"], my_string)

        if MHz:
            my_string = re.sub(u'([aAiIuUeEoO])([MHz])', lambda m: self.hashv_w2i[m.group(1)]+self.hashmd_w2i[m.group(2)], my_string)
        my_string = re.sub(u'([aAiIuUeEoO])', lambda m: self.hashv_w2i[m.group(1)], my_string)
        my_string = my_string.replace(u'.', self.hashc_w2i[u"."])
        return my_string
    
    def iscii2unicode(self, iscii):
        """Convert ISCII to Unicode"""
        if self.lang_tag in ["hin", "mar", "nep"]:
            unicode_ = self.iscii2unicode_hin(iscii)
        elif self.lang_tag == "tel":
            unicode_ = self.iscii2unicode_tel(iscii)
        elif self.lang_tag == "ben":
            unicode_ = self.iscii2unicode_ben(iscii)
        elif self.lang_tag == "kan":
            unicode_ = self.iscii2unicode_kan(iscii)
        elif self.lang_tag == "pan":
            unicode_ = self.iscii2unicode_pan(iscii)
        elif self.lang_tag == "mal":
            unicode_ = self.iscii2unicode_mal(iscii)
        elif self.lang_tag == "tam":
            unicode_ = self.iscii2unicode_tam(iscii)
        elif self.lang_tag == "ori":
            unicode_ = self.iscii2unicode_ori(iscii)
        else:
            sys.stderr.write("LanguageError: invalid language code\n")
            sys.exit(0)
        return unicode_
    
    def iscii2unicode_hin(self, iscii):
        unicode_ = self.i2u_h.sub(lambda m: self.hashh_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_tel(self, iscii):
        unicode_ = self.i2u_t.sub(lambda m: self.hasht_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_pan(self, iscii):
        unicode_ = self.i2u_p.sub(lambda m: self.hashp_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_kan(self, iscii):
        unicode_ = self.i2u_k.sub(lambda m: self.hashk_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_mal(self, iscii):
        unicode_ = self.i2u_m.sub(lambda m: self.hashm_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_ben(self, iscii): 
        unicode_ = self.i2u_b.sub(lambda m: self.hashb_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_tam(self, iscii):
        unicode_ = self.i2u_ta.sub(lambda m: self.hashcta_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def iscii2unicode_ori(self, iscii):
        unicode_ = self.i2u_o.sub(lambda m: self.hasho_i2u.get(m.group(1), u""), iscii)
        return unicode_
    
    def unicode2iscii(self, unicode_):
        """Convert Unicode to ISCII"""
        if self.lang_tag in ["hin", "mar", "nep"]:
            iscii = self.unicode2iscii_hin(unicode_)
        elif self.lang_tag == "tel":
            iscii = self.unicode2iscii_tel(unicode_)
        elif self.lang_tag == "ben":
            iscii = self.unicode2iscii_ben(unicode_)
        elif self.lang_tag == "kan":
            iscii = self.unicode2iscii_kan(unicode_)
        elif self.lang_tag == "pan":
            iscii = self.unicode2iscii_pan(unicode_)
        elif self.lang_tag == "mal":
            iscii = self.unicode2iscii_mal(unicode_)
        elif self.lang_tag == "tam":
            iscii = self.unicode2iscii_tam(unicode_)
        elif self.lang_tag == "ori":
            iscii = self.unicode2iscii_ori(unicode_)
        else:
            sys.stderr.write("LanguageError: invalid language code\n")
            sys.exit(0)
        return iscii
    
    def iscii2wx(self, my_string):
        """Convert ISCII to WX"""
        # CONSONANT+HALANT
        my_string = self.ch.sub(lambda m:self.hashc_i2w[m.group(1)], my_string)
        # CONSONANT+NUKTA+MATRA+MODIFIER 
        my_string = self.cnmmd.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashc_i2w[u"\xE9"]+
                    self.hashm_i2w[m.group(2)]+self.hashmd_i2w[m.group(3)], my_string)
        # CONSONANT+NUKTA+MATRA 
        my_string = self.cnm.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashc_i2w[u"\xE9"]+self.hashm_i2w[m.group(2)], my_string)
        # CONSONANT+NUKTA+MODIFIER 
        my_string = self.cnmd.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashc_i2w[u"\xE9"]+self.hashmd_i2w[m.group(2)], my_string)
        # CONSONANT+NUKTA+HALANT (added -Rashid 29-December-11 wanaKZvaxara)
        my_string = self.cnh.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashc_i2w[u"\xE9"], my_string)
        # CONSONANT+NUKTA 
        my_string = self.cn.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashc_i2w[u"\xE9"]+u"a", my_string)
        # CONSONANT+MATRA+MODIFIER 
        my_string = self.cmmd.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashm_i2w[m.group(2)]+self.hashmd_i2w[m.group(3)], my_string)
        # CONSONANT+MATRA 
        my_string = self.cm.sub(lambda m:self.hashc_i2w[m.group(1)]+self.hashm_i2w[m.group(2)], my_string)
        #CONSONANT+MODIFIER 
        my_string = self.cmd.sub(lambda m:self.hashc_i2w[m.group(1)]+u"a"+self.hashmd_i2w[m.group(2)], my_string)
        #CONSONANT 
        my_string = self.c.sub(lambda m:self.hashc_i2w[m.group(1)]+u"a", my_string)
        #VOWEL+MODIFIER, VOWEL, MATRA
        my_string = self.vmd.sub(lambda m:self.hashv_i2w[m.group(1)]+self.hashmd_i2w[m.group(2)], my_string)
        my_string = self.amd.sub(lambda m: u"a"+self.hashmd_i2w[m.group(1)], my_string)
        my_string = self.v.sub(lambda m:self.hashv_i2w[m.group(1)], my_string)
        #VOWEL A, FULL STOP or VIRAM Northern Scripts 
        my_string = my_string.replace(u"\xA4", u"a")
        my_string = my_string.replace(u"\xEA", u".")
        #For PUNJABI ADDAK 
        my_string = my_string.replace(u"\xFB", u"Y")
        #Replace ISCII Digits with Roman
        my_string = self.dig.sub(lambda m:self.digits_i2w[m.group(1)], my_string)
        return my_string
                                
    def unicode2iscii_hin(self, unicode_):
        # Normalize Unicode values (NUKTA variations)
        iscii_hin = self.u2i_hn.sub(lambda m:self.unicode_norm_hashh_u2i.get(m.group(1), u"")+u"\u093C", unicode_)
        # Convert Unicode values to ISCII values
        iscii_hin = self.u2i_h.sub(lambda m:self.hashh_u2i.get(m.group(1), u""), iscii_hin)
        return iscii_hin
    
    def unicode2iscii_tel(self, unicode_):
        # Convert Telugu Unicode values to ISCII values
        iscii_tel = self.u2i_t.sub(lambda m:self.hasht_u2i.get(m.group(1), u""), unicode_)
        return iscii_tel
    
    def unicode2iscii_pan(self, unicode_):
        # Normalize Unicode values (NUKTA variations)
        iscii_pan = self.u2i_pn.sub(lambda m:self.unicode_norm_hashp_u2i.get(m.group(1), u"")+u"\u0A3C", unicode_)
        #Convert Unicode values 0x0A5C to ISCII 
        iscii_pan = iscii_pan.replace(u"\u0A5C", u"\xBF\xE9")
        #Convert Unicode Devanagari VIRAM to ISCII
        iscii_pan = iscii_pan.replace(u"\u0964", u"\xEA")
        #Convert Unicode Punjabi values to ISCII values
        iscii_pan = self.u2i_p.sub(lambda m:self.hashp_u2i.get(m.group(1), u""), iscii_pan)
        return iscii_pan
    
    def unicode2iscii_kan(self, unicode_):
        # Normalize Unicode values (NUKTA variations), needs to be modified yet, PAWAN
        iscii_kan = self.u2i_kn.sub(lambda m:self.unicode_norm_hashk_u2i.get(m.group(1), u""), unicode_)
        # Convert Unicode values to ISCII values
        iscii_kan = self.u2i_k.sub(lambda m:self.hashk_u2i.get(m.group(1), u""), iscii_kan)
        return iscii_kan
    
    def unicode2iscii_mal(self, unicode_):
        # Convert Unicode values to ISCII values
        iscii_mal = self.u2i_m.sub(lambda m:self.hashm_u2i.get(m.group(1), u""), unicode_)
        return iscii_mal
    
    def unicode2iscii_ben(self, unicode_):
        # Convert Unicode values to ISCII values
        iscii_ben = self.u2i_b.sub(lambda m:self.hashb_u2i.get(m.group(1), u""), unicode_)
        return iscii_ben
    
    def unicode2iscii_tam(self, unicode_):
        # Convert Unicode values to ISCII values
        # Rashid added normalize case for tamil as per feedback
        # normalize two-part dependent vowel sign o ொ 
        iscii_tam = unicode_.replace(u"\u0BC6\u0BBE",u"\xE4")
        # normalize two-part dependent vowel sign oo ோ 
        iscii_tam = iscii_tam.replace(u"\u0BC7\u0BBE", u"\xE5")
        iscii_tam = self.u2i_ta.sub(lambda m:self.hashta_u2i.get(m.group(1), u"") , iscii_tam)
        return iscii_tam
    
    def unicode2iscii_ori(self, unicode_):
        # Normalize Unicode values (NUKTA variations)
        # Convert Unicode values to ISCII values
        iscii_ori = self.u2i_o.sub(lambda m:self.hasho_u2i.get(m.group(1), u""), unicode_)
        return iscii_ori
    
    def utf2wx(self, unicode_, tag):
        """Convert UTF-8 string to Unicode"""
        self.lang_tag = tag
        if not isinstance(unicode_, unicode):
            unicode_ = unicode_.decode('utf-8')
        unicode_ = self.normalize(unicode_)
        #Convert Unicode values with ISCII values
        iscii = self.unicode2iscii(unicode_)
        #Convert ISCII to WX-Roman
        wx = self.iscii2wx(iscii)
        #NOTE Consecutive Vowel Normalization
        wx = re.sub(u'[\xA0-\xFA]+', u'', wx)
        return wx.encode('utf-8')
    
    def wx2utf(self, wx, tag):
        """Convert WX-Roman to ISCII"""
        self.lang_tag = tag
        if not isinstance(wx, unicode):
            wx = wx.decode('utf-8')
        #NOTE Map iscii characters (if any) to some highly unlikely strings
        wx = self.isc.sub(lambda m: self.iscii_num[m.group(1)], wx)
        iscii = self.wx2iscii(wx)
        # Convert ISCII to Unicode
        unicode_ = self.iscii2unicode(iscii)
        # Marathi uses "." instead of "danda"
        if self.lang_tag == "mar":
            unicode_ = unicode_.replace(u"\u0964", u".")
        #NOTE Convert back the mapped iscii characters
        unicode_ = self.num.sub(lambda m: self.num_iscii[m.group(1)], unicode_)
        #Convert Unicode to utf-8
        return unicode_.encode('utf-8')
