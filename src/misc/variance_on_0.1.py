import numpy as np

if __name__ == "__main__":
    vary_lab = np.load("eval_data/dmt_loss.npy")
    vary_dms = np.load("eval_data/dmt_loss_dms.npy")
    vary_epoch = np.load("eval_data/dmt_loss_epoch.npy")
    vary_plabel = np.load("eval_data/plabel_loss.npy")

    default_plabel = np.mean(vary_plabel)
    ste_plabel = np.std(vary_plabel) / np.sqrt(5)
    defaut_lab = vary_lab[3]
    default_dms = vary_dms[2]
    defalt_epoch = vary_epoch[2]

    print(f"mean: {np.mean([defaut_lab, default_dms, defalt_epoch]):.3f}")
    print(f"ste: { np.std([defaut_lab, default_dms, defalt_epoch])/np.sqrt(3):.4f}")

    print("=====================================")
    print(f"Plab: {default_plabel:.3f}")
    print(f"ste: {ste_plabel:.4f}")

    print("=====================================")
    baselines = np.load("eval_data/baseline_loss.npy")
    print(f"upper bound: {np.mean(baselines.reshape(7,5)[-1]):.3f}")
    print(f"ste: { np.std(baselines.reshape(7,5 )[ -1])/np.sqrt(5):.4f}")
    print(f"lower bound: {np.mean(baselines.reshape(7,5)[3]):4f}")
    print(f"ste: { np.std(baselines.reshape(7, 5)[3])/np.sqrt(5):.4f}")
