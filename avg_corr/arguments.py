def classic(buffer,random_weight):
    match random_weight:
        case 0.3:
            match buffer:
                case 40:
                    batch = 256
                    link = 'identity'
                    alpha = 0.01
                    lr = 0.005
                    loss = 'mse'
                case 80:
                    batch = 256
                    link = 'identity'
                    alpha = 0.002
                    lr = 0.0001
                    loss = 'mse'
                case 200:
                    batch = 256
                    link = 'identity'
                    alpha = 0.01
                    lr = 0.0001
                    loss = 'mse'
        case 0.5:
            match buffer:
                case 40:
                    batch = 512
                    link = 'identity'
                    alpha = 0.002
                    lr = 0.005
                    loss = 'mse'
                case 80:
                    batch = 256
                    link = 'inverse'
                    alpha = 0.0005
                    lr = 0.001
                    loss = 'mse'
                case 200:
                    batch = 512
                    link = 'identity'
                    alpha = 0.0005
                    lr = 0.005
                    loss = 'mse'
        case 0.7:
            match buffer:
                case 40:
                    batch = 512
                    link = 'identity'
                    alpha = 0.0005
                    lr = 0.0005
                    loss = 'mse'
                case 80:
                    batch = 512
                    link = 'identity'
                    alpha = 0.001
                    lr = 0.0005
                    loss = 'mse'
                case 200:
                    batch = 512
                    link = 'inverse'
                    alpha = 0.01
                    lr = 0.001
                    loss = 'mse'
    return batch, link, alpha,lr, loss
