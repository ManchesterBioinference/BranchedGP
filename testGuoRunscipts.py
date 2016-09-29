import os

geneList = ['Actb', 'Ahcy', 'Aqp3', 'Atp12a', 'Bmp4', 'Cdx2', 'Creb312', 'Cebpa',
            'Dab2', 'DppaI', 'Eomes', 'Esrrb', 'Fgf4', 'Fgfr2', 'Fn1', 'Gapdh',
            'Gata3', 'Gata4', 'Gata6', 'Grhl1', 'Grhl2', 'Hand1', 'Hnf4a', 'Id2',
            'Klf2', 'Klf4', 'Klf5', 'Krt8', 'Lcp1', 'Mbnl3', 'Msc', 'Msx2', 'Nanog',
            'Pdgfa', 'Pdgfra', 'Pecam1', 'Pou5f1', 'Runx1', 'Sox2', 'Sall4',
            'Sox17', 'Snail', 'Sox13', 'Tcfap2a', 'Tcfap2c', 'Tcf23', 'Utf1',
            'Tspan8']


for g in geneList:
    runcmd = "ipython testGuoSingleArg.py %s" % g
    print('Running', runcmd)
    os.system(runcmd)
