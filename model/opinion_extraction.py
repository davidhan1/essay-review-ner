from pyhanlp import JClass
import operator
def parseSentWithKey(sentence, key=None):
    #key是关键字，如果关键字存在，则只分析存在关键词key的句子，如果没有key，则不判断。
    if key:
        keyIndex = 0
        if key not in sentence:
            return []
    rootList = []
    hanlp = JClass('com.hankcs.hanlp.HanLP')
    parse_result = str(hanlp.parseDependency(sentence)).strip().split('\n')
    print(hanlp.parseDependency(sentence))
    # 索引-1，改正确，因为从pyhanlp出来的索引是从1开始的。
    for i in range(len(parse_result)):
        parse_result[i] = parse_result[i].split('\t')
        parse_result[i][0] = int(parse_result[i][0]) - 1
        parse_result[i][6] = int(parse_result[i][6]) - 1
        if key and parse_result[i][1] == key:
            keyIndex = i
    main_relation = set(['核心关系'])
    for i in range(len(parse_result)):
        self_index = int(parse_result[i][0])
        target_index = int(parse_result[i][6])
        relation = parse_result[i][7]
        if relation in main_relation:
            if self_index not in rootList:
                rootList.append(self_index)
        # 寻找多个root，和root是并列关系的也是root
        elif relation == "并列关系" and target_index in rootList:
            if self_index not in rootList:
                rootList.append(self_index)


        if len(parse_result[target_index]) == 10:
            parse_result[target_index].append([])

        #对依存关系，再加一个第11项，第11项是一个当前这个依存关系指向的其他索引
        if target_index != -1 and not (relation == "并列关系" and target_index in rootList):
            parse_result[target_index][10].append(self_index)
    
    # 寻找key在的那一条root路径
    if key:
        rootIndex = 0
        if len(rootList) > 1:
            target = keyIndex
            while True:
                if target in rootList:
                    rootIndex = rootList.index(target)
                    break
                next_item = parse_result[target]
                target = int(next_item[6])
        loopRoot = [rootList[rootIndex]]
    else:
        loopRoot = rootList

    result = {}
    related_words = set()
    for root in loopRoot:
        # 把key和root加入到result中
        if key:
            addToResult(parse_result, keyIndex, result, related_words)
        addToResult(parse_result, root, result, related_words)

    reverse_relation = set(['动补结构', '动宾关系', '介宾关系', '状中结构'])
    #根据'动补结构', '动宾关系', '介宾关系'，选择观点
    for item in parse_result:
        relation = item[7]
        target = int(item[6])
        index = int(item[0])
        if relation in reverse_relation and target in result and target not in related_words:
            addToResult(parse_result, index, result, related_words)

    # 加入关键词
    for item in parse_result:
        word = item[1]
        if word == key:
            result[int(item[0])] = word

    #对已经在result中的词，按照在句子中原来的顺序排列
    sorted_keys = sorted(result.items(), key=operator.itemgetter(0))
    selected_words = [w[1] for w in sorted_keys]
    return selected_words

def addToResult(parse_result, index, result, related_words):
    result[index] = parse_result[index][1]
    if len(parse_result[index]) == 10:
        return
    reverse_target_index = 0
    for i in parse_result[index][10]:
        if i < index and i > reverse_target_index:
            reverse_target_index = i
    if abs(index - reverse_target_index) <= 1:
        result[reverse_target_index] = parse_result[reverse_target_index][1]
        related_words.add(reverse_target_index)

print(parseSentWithKey('对书房的描写以及阅读感受的叙述很见功底，文字优美，有才情', key=None))