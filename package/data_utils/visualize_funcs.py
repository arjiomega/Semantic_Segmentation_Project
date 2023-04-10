import matplotlib.pyplot as plt

def visualize_class_distribution(img_list:list[str],img_label:dict[str,int],map_label:dict[int,str]=None,
                                 title:str='Class Count Comparison',
                                 xlabel:str='Species',
                                 ylabel:str='Count',
                                 figsize:tuple[int,int]=None):

    if map_label:
        assert set(img_label.values()) == set(map_label.keys()), "incomplete map_label"

    classes = set(img_label.values())
    class_count = [len([1 for img in img_list if img.split('.')[0] in img_label and img_label[img.split('.')[0]] == class_ ]) for class_ in classes]

    class_names = [map_label[label] for label in classes] if map_label else list(classes)

    if figsize:
        plt.figure(figsize=figsize)

    plt.bar(class_names,class_count)
    plt.xticks(class_names,rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()