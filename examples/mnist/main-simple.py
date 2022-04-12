import torch
import torch.optim as optim
from sketch import SketchAlgorithm
import torch.nn.functional as F
import logging
import bagua.torch_api as bagua

def main():
    torch.manual_seed(42)
    torch.cuda.set_device(bagua.get_local_rank())
    bagua.init_process_group()
    logging.getLogger().setLevel(logging.INFO)

    model = torch.nn.Sequential(torch.nn.Linear(200, 1)).cuda()

    for m in model.modules():
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.do_sketching = False


    optimizer = optim.SGD(model.parameters(), lr=1)
    algorithm = SketchAlgorithm(optimizer)

    model = model.with_bagua(
        [optimizer],
        algorithm,
        do_flatten=True,
    )

    model.train()
    X = torch.randn(1000, 200).cuda()
    y = torch.zeros(1000, 1).cuda()

    for epoch in range(1, 101):
        optimizer.zero_grad()
        output = model(X)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        logging.info(f"it {epoch}, loss: {loss.item():.6f}")

        

if __name__ == "__main__":
    main()