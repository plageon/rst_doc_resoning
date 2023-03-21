import amrlib
import spacy
amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test of the SpaCy extension. The test has multiple sentences.')

# The following are roughly equivalent but demonstrate the different objects.
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)

for span in doc.sents:
    graphs = span._.to_amr()
    print(graphs[0])