import torch
import torch.nn as nn
import tensorboardX
import torchvision
from vggnet import VGGNet
from mobilenet_v1 import mobilenet_v1_small
from resnet import resnet
from inceptionMouble import InceptionNetSmall
from load_cifar10 import train_loader, test_loader
import os

# models save path
models_path = './models/InceptionNetSmall'
log_path = './log/InceptionNetSmall'
# if has GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 200
lr = 0.01
batch_size = 128
net = InceptionNetSmall().to(device)

 #loss
loss_fun = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# weight_decay : SGD 正则项
# optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=5e-4)

# lr decrease after every 5 epoch, rate = gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


if __name__ == '__main__':

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = tensorboardX.SummaryWriter(log_path)

    step_n = 0
    for epoch in range(epoch_num):
        net.train() #train BN dropout

        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_fun(outputs, labels)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs.data,dim=1)

            correct = pred.eq(labels.data).cpu().sum()

            # print("epoch is ", epoch)
            # print("step ,", i," loss is: ", loss.item(),
            #       "mini-batch correct is :", 100.0 * correct.item() / batch_size)
            # print("train lr is ", optimizer.state_dict()['param_groups'][0]['lr'])

            writer.add_scalar("train loss", loss.item(),global_step=step_n)
            writer.add_scalar("train correct", 100.0 * correct / batch_size, global_step=step_n)

            im = torchvision.utils.make_grid(inputs)
            writer.add_image("train im", im, global_step=step_n)
            step_n += 1
            if not os.path.exists(models_path):
                os.mkdir(models_path)
            torch.save(net.state_dict(), "models/{}.pth".format(epoch + 1))

            scheduler.step()

        sum_loss = 0.0
        sum_correct = 0.0
        for i, data in enumerate(test_loader):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_fun(outputs, labels)
            _, pred = torch.max(outputs.data,dim=1)

            correct = pred.eq(labels.data).cpu().sum()

            sum_loss += loss.item()
            correct += correct.item()

            im = torchvision.utils.make_grid(inputs)
            writer.add_image("test im", im, global_step=step_n)
        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 100.0 / len(test_loader) / batch_size

        writer.add_scalar("test loss", test_loss, global_step=epoch + 1)
        writer.add_scalar("test correct", test_correct, global_step=epoch + 1)
        print("test step ,", i," loss is: ", test_loss,
              "test_correct is :", test_correct)
        print("test lr is ", optimizer.state_dict()['param_groups'][0]['lr'])

    writer.close()