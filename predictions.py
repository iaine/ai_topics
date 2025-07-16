import json

#@todo: provide the file and tag as command line?
with open(filename, 'r') as fh:
    data json.load(fh.read())
fh.closed()

data = numpy.array([numpy.array(xi) for xi in x])

tag = ""
tagpos = data["labels"].index(tag)

data[:, tagpos]

#get number of segments and then parse into time
num_segments = len(data["scores"])

#get audio length and divide by segments
audio_files = []
#get pre-existing tags
classes = list(yamntags)
ground_truth = [tag]


prompt = 'this is a sound of '
class_prompts = [prompt + x for x in classes]

# Load and initialize CLAP
# Setting use_cuda = True will load the model on a GPU using CUDA
clap_model = CLAP(version = '2023', use_cuda=False)

# compute text embeddings from natural text
text_embeddings = clap_model.get_text_embeddings(class_prompts)

for audio_segment in audio_files:

    # compute the audio embeddings from an audio file
    audio_embeddings = clap_model.get_audio_embeddings(audio_files, resample=False)

    # compute the similarity between audio_embeddings and text_embeddings
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    similarity = F.softmax(similarity, dim=1)
    values, indices = similarity[0].topk(10)

    # Print the results
    print("Ground Truth: {}".format(ground_truth))
    print("Top predictions:\n")
    for value, index in zip(values, indices):
        print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
