import torch
import torchaudio

knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device='cuda')


def modulate(src, ref, out):
    """Modulates the source wav file to ref wav file. Both the files should be at 16kHz"""
    query_seq = knn_vc.get_features(src)
    matching_set = knn_vc.get_matching_set(ref)

    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    torchaudio.save(out, out_wav[None], 16000)
