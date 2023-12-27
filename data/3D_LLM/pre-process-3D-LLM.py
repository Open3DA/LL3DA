import json

mapping_file = json.load(open('final_scene_map_dict_scan_v2.json', 'r'))
data = json.load(open('data_part2_scene.json', 'r'))

scannet_annotations = []

corpus = set()

for anno in data:
    scannet_scene_id = mapping_file.get(str(anno['scene_id']), None)
    if scannet_scene_id is None:
        continue
    anno['scene_id'] = scannet_scene_id
    anno['question'] = anno['question'].replace('Robot:', ' ### assistant:')
    anno['question'] = anno['question'].replace('Human:', ' ### human:')
    anno['question'] = anno['question'].replace('Agent 1:', ' ### human:')
    anno['question'] = anno['question'].replace('Agent 2:', ' ### assistant:')
    anno['question'] = anno['question'].replace('Agent1:', ' ### human:')
    anno['question'] = anno['question'].replace('Agent2:', ' ### assistant:')
    anno['question'] = anno['question'].replace('Agent A:', ' ### human:')
    anno['question'] = anno['question'].replace('Agent B:', ' ### assistant:')
    anno['question'] = anno['question'].strip()
    
    anno['question'] = '### human: ' + anno['question'] + ' ### assistant:'
    anno['question'] = anno['question'].replace('### human: ### human: ', '### human: ')
    anno['question'] = anno['question'].replace('### assistant: ### assistant:', '### assistant:')
    anno['question'] = anno['question'].replace('  ', ' ')
    
    if 'Agent 1:' in anno['answers'][0]:
        continue
    if 'Agent A:' in anno['answers'][0]:
        continue
    if 'Agent1:' in anno['answers'][0]:
        continue
    
    anno['answers'][0] = anno['answers'][0].replace('Agent 2:', ' ')
    anno['answers'][0] = anno['answers'][0].replace('Agent2:', ' ')
    anno['answers'][0] = anno['answers'][0].replace('Agent B:', ' ')
    anno['answers'][0] = anno['answers'][0].replace('Robot:', ' ')
    anno['answers'][0] = anno['answers'][0].strip()
    
    if anno['answers'][0] == 'Agent':
        continue
    if anno['answers'][0] == 'Agent 1':
        continue
    if anno['answers'][0] == 'Agent 2':
        continue
    if anno['answers'][0] == 'Robot':
        continue
    if anno['answers'][0] == '':
        continue
    
    value = 'question: ' + anno['question'] + ' answer: ' + anno['answers'][0]
    if value in corpus:
        continue
    corpus.add(value)
    scannet_annotations.append(anno)

with open('3d_llm_scannet_data.json', 'w') as file:
    json.dump(scannet_annotations, file, indent=4)