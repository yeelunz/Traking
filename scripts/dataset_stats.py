import os
base=r"C:\Users\User\Desktop\code\Traking\dataset\merged"

disease_dirs=[]
normal_dirs=[]
for name in os.listdir(base):
    path=os.path.join(base,name)
    if os.path.isdir(path):
        if name.startswith('n'):
            normal_dirs.append(path)
        else:
            disease_dirs.append(path)

print('disease patients', len(disease_dirs))
print('normal patients', len(normal_dirs))

stats={'disease':{}, 'normal':{}}
def process_group(dirs, key):
    total_frames=0
    modality_counts={}
    for d in dirs:
        seg=os.path.join(d,'seg_masks')
        if not os.path.isdir(seg): continue
        for mod in os.listdir(seg):
            modpath=os.path.join(seg,mod)
            if os.path.isdir(modpath):
                n=len([f for f in os.listdir(modpath) if os.path.isfile(os.path.join(modpath,f))])
                modality_counts[mod]=modality_counts.get(mod,0)+n
                total_frames+=n
    stats[key]['total_frames']=total_frames
    stats[key]['modalities']=modality_counts

process_group(disease_dirs,'disease')
process_group(normal_dirs,'normal')
print(stats)
