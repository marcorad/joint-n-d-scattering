PATH = 'hsi-results/'
import os
import json

for fname in os.listdir(PATH):
    print('------------')
    print(fname)
    print('------------')
    with open(PATH + fname) as file:
        res = json.load(file)
        best_15 = res[0]
        best_10perc = res[1]
        for r in res[1:]:
            if r['ntrain'] == 15:
                if r['mean'] > best_15['mean']:
                    best_15 = r
            else:
                if r['mean'] > best_10perc['mean']:
                    best_10perc = r
        print(best_15)
        print(best_10perc)
            
    print()
        