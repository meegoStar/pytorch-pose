def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, prediction = output.topk(maxk, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))

    return result
