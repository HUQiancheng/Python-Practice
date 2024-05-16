def summarize_model(model):
    print(model)
    print("Total parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
