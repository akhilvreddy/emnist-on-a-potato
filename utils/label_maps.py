def get_label_map(dataset_name):
    """
    Returns the label map for a given dataset.
    
    Args:
        dataset_name (str): One of ["mnist", "emnist_balanced"]
    
    Returns:
        list of str: Label map
    """
    
    if dataset_name == "mnist":
        return [str(i) for i in range(10)]

    elif dataset_name == "emnist_balanced":
        return [
            '0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','I','J',
            'K','L','M','N','O','P','Q','R','S','T',
            'U','V','W','X','Y','Z',
            'a','b','d','e','f','g','h','n','q','r','t'
        ]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")