import torch
import numpy as np
from itertools import product
import networkx as nx
import torch.nn.functional as F
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from sklearn.cluster import SpectralClustering
from einops import rearrange


def Generate_Graph_ST(mask_argmax_tensor):
    """
    mask_argmax_tensor: tensor [T, W, H]
    """
    # Find All connect mask
    mask_np = mask_argmax_tensor.detach().cpu().numpy()
    
    t = mask_np.shape[0]
    m = mask_np.shape[1]
    n = mask_np.shape[2]
    edges = []
    for k, i, j in product(range(t), range(m), range(n)):
        for delta_k, delta_i, delta_j in product([0, 1], [0, 1], [0, 1]):
            k_neighbor = min(k + delta_k, t - 1) 
            i_neighbor = min(i + delta_i, m - 1) 
            j_neighbor = min(j + delta_j, n - 1) 
            edges.append((mask_np[k][i][j], mask_np[k_neighbor][i_neighbor][j_neighbor]))

    edges = list(set(edges))
    edges = [(i, j) for i, j in edges if i != j]
    edges = list(set(edges))

    # genetage mask graph
    cluster_ids = np.sort(np.unique(mask_np.reshape(-1)))
    G = nx.Graph()
    for i in cluster_ids:
        G.add_node(i, weight=mask_argmax_tensor.eq(i).float().reshape(-1, 1))       # [THW, 1] bool tensor

    for i, j in edges:
        G.add_edge(i, j)

    return G

############################
#           merging        #
############################
def merge_mask(G, patch_feat, prompts, criterion=None):
    # every node in G have a weight denoting the mask of the slot
    while True:
        have_contractable = False
        for i in G.nodes():
            # Node Traversal
            inner_break = False
            for j in G.neighbors(i):
                # Neighborhood Traversal
                if criterion(G, i, j, patch_feat, prompts):
                    mask1 = G.nodes[i]["weight"]
                    mask2 = G.nodes[j]["weight"]
                    G = nx.contracted_nodes(G, i, j, self_loops=False)
                    G.nodes[i]["weight"] = mask1 + mask2
                    have_contractable = True
                    inner_break = True
                    break
            if inner_break:
                break
        
        if not have_contractable:
            break
    return torch.stack([G.nodes[i]["weight"].reshape(-1) for i in G.nodes]), G     # [K, num_patch]


def merge_criterion(G, i, j, patch_feat, prompts):
    patch_text_match = patch_feat @ prompts.text_features.permute(1, 0)     # [L, D]x[D, num_prompts]->[L, num_prompts]
    logits_i = (G.nodes[i]["weight"].permute(1, 0) @ patch_text_match).reshape(-1)  # [1, L] x [L, num_prompts] -> [1, num_prompts]
    logits_j = (G.nodes[j]["weight"].permute(1, 0) @ patch_text_match).reshape(-1)
    merged_logits = logits_i + logits_j
    ind_i = logits_i.argmax(-1)
    ind_j = logits_j.argmax(-1)
    merged_indx = merged_logits.argmax(-1)
    
    if prompts.is_same_class(ind_i, ind_j) and prompts.is_same_class(ind_i, merged_indx):
        return True
    else:
        return False

############################
#         mask refine      #
############################
def setwise_distance(a, b=None):
    if b is None:
        b = a
    return torch.pow((a.unsqueeze(dim=1) - b.unsqueeze(dim=0)), 2.0).sum(dim=-1)


def kmeans(x, n_clusters, n_init=1, tol=1e-4):
    best_loss = None
    best_result = None
    best_state = None
    
    for init in range(n_init):
        idx = np.random.choice(x.shape[0], n_clusters)
        state = x[idx]
        while True:
            pre_state = state.clone()
            dis = setwise_distance(x, state).squeeze()
            result = torch.argmin(dis, dim=1)

            for i in range(n_clusters):
                idx = torch.nonzero(result == i).squeeze()
                items = torch.index_select(x, 0, idx)
                if items.size(0):
                    state[i] = items.mean(dim=0)
                else:
                    state[i] = pre_state[i].clone()

            shift = torch.pairwise_distance(pre_state, state)
            total = torch.pow(torch.sum(shift), 2.0)

            if total < tol:
                row_ind = torch.arange(dis.shape[0], device=dis.device)
                loss = dis[row_ind, result].sum()
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_result = result
                    best_state = state
                break

    return best_result, best_state, best_loss


def spectral_clustering(n_clusters, adj, n_init=10, tol=1e-4):
    diag = adj.sum(1).diag()
    laplican = diag - adj

    inv_diag = torch.diag(torch.pow(torch.diag(diag), -0.5))
    sym_laplican = inv_diag.mm(laplican).mm(inv_diag)

    e, v = torch.lobpcg(sym_laplican, k=n_clusters, largest=False)#, method="ortho")

    norm_v = v.div(v.norm(p=2, dim=1, keepdim=True).expand_as(v) + tol)
    
    result, state, loss = kmeans(norm_v, n_clusters, n_init, tol)

    return result, loss


def spectral_clustering_sklearn(n_clusters, adj_np):
    return SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30).fit(adj_np).labels_, 0


def spectral_clustering_dynamic(n_clusters_min, n_clusters_max, adj, sc_fn):
    masks = []
    losses = []
    for n_clusters in range(n_clusters_min, n_clusters_max+1):
        try:
            mask, loss = sc_fn(n_clusters, adj)
            masks.append(mask)
            losses.append(loss)
        except Exception as e:
            pass
    if len(losses) == 0:
        raise ValueError(f"All cluster sizes failed: {e}")
    return masks[min(enumerate(losses), key=lambda x: x[1])[0]]


def sc_refinement(mask, n_spectral_clustering, refine_masks):
    mask = mask.argmax(dim=2)
    refine_masks = refine_masks.lower()
    st_mask = rearrange(mask, 's t p -> s (t p)').T # (8x2800) -> [6,5,3,6,0,1,2,5,6,2]
    n_patch = st_mask.shape[0]
    n_mask = st_mask.shape[1]
    n_slot = st_mask.max().item()+1

    col = (torch.arange(n_mask, device=st_mask.device)*n_slot).unsqueeze(0) + st_mask
    row = torch.arange(n_patch, device=st_mask.device).unsqueeze(1).repeat(1, n_mask) 
    node_hot_feat = torch.zeros((n_patch, n_mask * n_slot), device=st_mask.device)
    node_hot_feat[row, col]=1 # [0,0,0,0,0,0,1, 0,0,0,0,0,1,0]

    adj = (node_hot_feat @ node_hot_feat.T) / n_mask
    assert len(n_spectral_clustering) == 2, n_spectral_clustering

    if (not n_spectral_clustering[1] == n_spectral_clustering[0]) and (not refine_masks == "gpu"):
        raise ValueError("Automatic n_cluster detection only supported on GPU (for now)")

    if refine_masks == "cpu":
        adj = adj.cpu().numpy()
        sc_fn = lambda n_cl, adj: spectral_clustering_sklearn(n_cl, adj)
    else:
        sc_fn = lambda n_cl, adj: spectral_clustering(n_clusters=n_cl, adj=adj, n_init=30)
    refine_mask = spectral_clustering_dynamic(*n_spectral_clustering, adj, sc_fn)
    refine_mask = rearrange(refine_mask, '(t p) -> t p', t=mask.shape[1])
    if refine_masks == "cpu":
        mask = torch.tensor(refine_mask, requires_grad=False, device=mask.device)
    else:
        mask = refine_mask
    return mask


############################
#         filtering        #
############################

def foreground_filter_criterion1(logits, prompts):
    """
    Need the idx of template of all foreground classes
    Keep all mask with maximum logit larger than 0
    mask_one_hot_idx: [K, HW]
    logits: [K, prompts_number]
    """
    threthold = 0
    foreground_logits = logits[:, prompts.return_foreground_idx()].max(-1)[0]
    index_1 = foreground_logits > threthold
    return index_1.detach().cpu().numpy()


def foreground_filter_criterion2(logits, prompts):
    """
    Need to judge if a idx is the foreground classes
    Keep all mask classified as COCO VID classes
    mask_one_hot_idx: [K, HW]
    logits: [K, prompts_number]
    """
    logits_argmax = logits.detach().cpu().numpy().argmax(-1)
    index_2 = prompts.is_foreground_idx(logits_argmax)
    return index_2


def foreground_filter_criterion3(logits, prompts):
    """
    combiend criterion1 and criterion2
    mask_one_hot_idx: [K, HW]
    logits: [K, prompts_number]
    """
    index_1 = foreground_filter_criterion1(logits, prompts)
    index_2 = foreground_filter_criterion2(logits, prompts)
    return index_1 & index_2

def background_filter_keep_one(clusters_idx, logits, prompts, criterion=foreground_filter_criterion3):
    """
    clusters_idx: [K]
    logits: [K, prompts_number]
    Keep at one slot in every frame
    """

    idx = criterion(logits, prompts)
    if np.sum(idx) > 0:
        return clusters_idx[~idx]
    else:
        # print("No Object Found! Force to have one!")
        MaxLogits = logits[:, prompts.return_foreground_idx()].max(-1)[0]
        MaxMaxLogitsIdx = MaxLogits.max(-1)[1].item()
        idx = (np.arange(len(logits)) == MaxMaxLogitsIdx)
        return clusters_idx[~idx]


def merge_mask_id(G, *args, **kargs):
    return torch.stack([G.nodes[i]["weight"].reshape(-1) for i in G.nodes])  # [K, num_patch]


def filtering(G, clip_pacl, image, mask_one_hot_idx, prompts, clusters_idx, patch_feat):
    logits = clip_pacl.average_patch_video(image, mask_one_hot_idx, prompts.text_features, patch_feat=patch_feat)  # [K, num_prompts]
    background_idx = background_filter_keep_one(clusters_idx, logits, prompts)

    # Delete all background slots
    for idx in background_idx:
        G.remove_node(idx)

    foreground_mask_one_hot_idx = merge_mask_id(G)      # [filtered_K, THW]

    return G, foreground_mask_one_hot_idx, background_idx


############################
#         matching         #
############################
def visual_text_matching_ST(image, mask,
                            prompts, foreground_prompts,
                            clip_pacl,
                            refine_masks, n_spectral_clustering,
                            filter_bg, merge_fg,
                            duplicate_box):
    """
    image: [T, H, W, C]
    mask: [n_stmae_seeds, T, K, num_patches]
    """
    # save some shape stuff
    T, H, W, C = image.shape
    NH, NW = H//16, W//16

    # refine mask by SC
    if refine_masks.lower() in ["cpu", "gpu"]:
        mask = sc_refinement(mask, n_spectral_clustering, refine_masks)
    else:
        mask = mask[0].argmax(dim=1)        # [T, HW]

    # run one forward of PACL to get patch features
    with torch.no_grad():
        _, _, _, _, patch_feat, _ = clip_pacl(image.permute(0, 3, 1, 2))    # [T, H, W, C] -> [T, HW, D]
        patch_feat = rearrange(patch_feat, 'T L D -> (T L) D')        # [THW, D]

    # build a networkX graph of slots for later background filtering and foreground merging
    mask_argmax = rearrange(mask, 'T (H W) -> T H W', H=NH, W=NW)            # [T, H, W]
    G = Generate_Graph_ST(mask_argmax)
    clusters_idx = np.array(G.nodes)
    n_objects = len(clusters_idx)

    clusters = torch.tensor(clusters_idx)[:, None, None, None].to(mask_argmax)      # [K, 1, 1, 1]
    mask_one_hot = (mask_argmax == clusters).long().to(mask.device)                 # [K, T, H, W]
    mask_one_hot_idx = mask_one_hot.view(n_objects, -1).to(torch.float)             # [K, THW]

    # background filtering
    if filter_bg:
        clusters_idx = np.array(G.nodes)
        G, mask_one_hot_idx, background_idx = filtering(G, clip_pacl, image, mask_one_hot_idx,
                                                        prompts, clusters_idx, patch_feat)   # [filtered_K, THW]
        # If all slots are background     seems has no effect since force-to-have-one strategy
        # if len(background_idx) == n_objects:
        #     merged_mask_one_hot_idx = None
        #     category_predictions = None
        #     return merged_mask_one_hot_idx, category_predictions
    foreground_mask_as_image_one_hot = F.interpolate(rearrange(mask_one_hot_idx, 'K (T H W) -> T K H W', T=T, H=NH, W=NW),
                                                     size=(H, W), mode='nearest-exact')

    # foreground merging
    if merge_fg:
        mask_one_hot_idx, G = merge_mask(G, patch_feat, foreground_prompts, merge_criterion)  # [K_merged, THW]

    # inference bounding boxs from masks
    n_objects_after_post_processing = mask_one_hot_idx.shape[0]
    logits_after_post_processing = clip_pacl.average_patch_video(image, mask_one_hot_idx,
                                                                 foreground_prompts.text_features,
                                                                 patch_feat=patch_feat)  # [K_merged, num_prompts]
    category_predictions = []
    if duplicate_box:
        # keep each categories' prediction confidence
        for idx in range(n_objects_after_post_processing):
            prob = (foreground_prompts.logit_of_classes(logits_after_post_processing[idx])).softmax(0)
            temp = []
            for c in range(int(prob.shape[0])):
                temp.append([c, foreground_prompts.classes_name[c], prob[c].item()])
            category_predictions.append(temp)
    else:
        for idx in range(n_objects_after_post_processing):
            prob = (foreground_prompts.logit_of_classes(logits_after_post_processing[idx])).softmax(0)
            text_ind = prob.argmax().detach().cpu().numpy().astype(np.int8)
            category_predictions.append([text_ind, foreground_prompts.classes_name[text_ind], prob.max().item()])

    # prepare some visualization
    mask_one_hot = rearrange(mask_one_hot_idx, 'K (T H W) -> T K H W', T=T, H=NH, W=NW)
    merged_mask_as_image_one_hot = F.interpolate(mask_one_hot, size=(H, W), mode='nearest-exact')

    mask_argmax_flat = mask_argmax.flatten(1, 2).unsqueeze(-1)
    mask_as_image_one_hot = mask_argmax_flat == mask_argmax_flat.unique()
    mask_as_image_one_hot = rearrange(mask_as_image_one_hot, "f (h w) n -> f n h w", h=NH, w=NW)
    mask_as_image_one_hot = F.interpolate(mask_as_image_one_hot.float(), size=(H, W), mode='nearest-exact')

    return foreground_mask_as_image_one_hot, merged_mask_as_image_one_hot, category_predictions, mask_as_image_one_hot



############################
#         matching         #
############################
def visual_text_matching_Spatial(image, mask,
                            prompts, foreground_prompts,
                            clip_pacl,
                            refine_masks, n_spectral_clustering,
                            filter_bg, merge_fg,
                            duplicate_box):
    """
    image: [H, W, C]
    mask: [n_stmae_seeds, K, num_patches]
    """
    image = image[None]
    mask = mask[:,None]
    return visual_text_matching_ST(image, mask,
                            prompts, foreground_prompts,
                            clip_pacl,
                            refine_masks, n_spectral_clustering,
                            filter_bg, merge_fg,
                            duplicate_box)
    