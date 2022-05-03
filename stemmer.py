suffixes = {
    1: [u"ो", u"े", u"ू", u"ु", u"ी", u"ि", u"ा"],
    2: [u"कर", u"ाओ", u"िए", u"ाई", u"ाए", u"ने", u"नी", u"ना", u"ते", u"ीं", u"ती", u"ता", u"ाँ", u"ां", u"ों", u"ें"],
    3: [u"ाकर", u"ाइए", u"ाईं", u"ाया", u"ेगी", u"ेगा", u"ोगी", u"ोगे", u"ाने", u"ाना", u"ाते", u"ाती", u"ाता", u"तीं", u"ाओं", u"ाएं", u"ुओं", u"ुएं", u"ुआं"],
    4: [u"ाएगी", u"ाएगा", u"ाओगी", u"ाओगे", u"एंगी", u"ेंगी", u"एंगे", u"ेंगे", u"ूंगी", u"ूंगा", u"ातीं", u"नाओं", u"नाएं", u"ताओं", u"ताएं", u"ियाँ", u"ियों", u"ियां"],
    5: [u"ाएंगी", u"ाएंगे", u"ाऊंगी", u"ाऊंगा", u"ाइयाँ", u"ाइयों", u"ाइयां"],
}


def stemmer(text):
    stems = ""
    words = text.split()
    for word in words:
        for L in range(1, 5):
            if len(word) >= L + 1:
                for suffix in suffixes[L]:
                    if word.endswith(suffix):
                        word = word[:-L]
        if word:
            stems = " ".join((stems, word)) if stems else word
    return stems
