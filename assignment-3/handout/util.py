import torch.autograd as autograd

def make_one_hot_vec(word,vo):
    t=torch.zeros(1,1,len(vo))
    t[0][0][vo[word]]=1
    return autograd.Variable(t)

def make_one_vec_target(word,vo):
    t=autograd.Variable(torch.LongTensor([vo[word]]))
    return t

def prepare_seq(seq,vo):
    idx=[vo[w] for w in seq]
    tensor=torch.LongTensor(idx)
    return autograd.Variable(tensor)

def toList(sen):
    res=[]
    for s in sen:
        res.append(s)
    return res

def makeforOneCase(s,voo):
    tmpIn=[]
    tmpout=[]
    for i in range(1,len(s)):
        w=s[i]
        w_b=s[i-1]
        tmpIn.append(voo[w_b])
        tmpOut.append(voo[w])
    return torch.cat(tmpIn),torch.cat(tmpOut)