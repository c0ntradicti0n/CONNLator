


import en_coref_sm
nlp = en_coref_sm.load()

doc = nlp(u'My sister has a dog. She loves him')

print (doc._.coref_clusters)
print (doc._.coref_clusters[1].mentions)
print (doc._.coref_clusters[1].mentions[-1])
print (doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main)

token = doc[-1]
print (token._.in_coref)
print (token._.coref_clusters)

span = doc[-1:]
print (span._.is_coref)
print (span._.coref_cluster.main)
print (span._.coref_cluster.main._.coref_cluster)