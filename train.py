import time
import os
from time import gmtime, strftime
from datetime import datetime
import torch
from torch import nn
import torch.utils.data
from torchfoldext import FoldExt
import util
from dynamicplot import DynamicPlot

from grassdata import GRASSDataset
from grassmodel import GRASSEncoder
from grassmodel import GRASSDecoder
import grassmodel


config = util.get_args()

config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)
if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA.")

encoder = GRASSEncoder(config)
decoder = GRASSDecoder(config)
if config.cuda:
    encoder.cuda()
    decoder.cuda()


print("Loading data ...... ", end='', flush=True)
grass_data = GRASSDataset(config.data_path)
def my_collate(batch): #自定义批处理拼接函数
    return batch
train_iter = torch.utils.data.DataLoader(grass_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
print("DONE")

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

print("Start training ...... ")

start = time.time()

if config.save_snapshot:
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    snapshot_folder = os.path.join(config.save_path, 'snapshots_'+strftime("%Y-%m-%d_%H-%M-%S",gmtime()))
    if not os.path.exists(snapshot_folder):
        os.makedirs(snapshot_folder)

if config.save_log:
    fd_log = open('training_log.log', mode='a')
    fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fd_log.write('\n#epoch: {}'.format(config.epochs))
    fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    fd_log.write('\ncuda: {}'.format(config.cuda))
    fd_log.flush()

header = '     Time    Epoch     Iteration    Progress(%)  ReconLoss  KLDivLoss  TotalLoss'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.2f},{:>10.2f},{:>10.2f}'.split(','))

total_iter = config.epochs * len(train_iter)

if not config.no_plot:
    plot_x = [x for x in range(total_iter)]
    plot_total_loss = [None for x in range(total_iter)]
    plot_geom_loss = [None for x in range(total_iter)]
    plot_cls_loss = [None for x in range(total_iter)]
    plot_sym_loss = [None for x in range(total_iter)]
    plot_cat_loss = [None for x in range(total_iter)]
    plot_kldiv_loss = [None for x in range(total_iter)]
    dyn_plot = DynamicPlot(title='Training loss over iterations (GRASS)', xdata=plot_x, ydata={'Total_loss':plot_total_loss, 'Geom_loss':plot_geom_loss, 'Cls_loss':plot_cls_loss, 'Sym_loss':plot_sym_loss, 'Cat_loss':plot_cat_loss, 'KLD_loss':plot_kldiv_loss})
    iter_id = 0
    max_loss = 0
    
header = '     Time    Epoch     Iteration    Progress(%)  GeomLoss   ClsLoss   SymLoss   CatLoss  KLDLoss  TotalLoss'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>8.2f},{:>8.2f},{:>8.2f},{:>8.2f},{:>8.2f},{:>10.2f}'.split(','))

for epoch in range(config.epochs):
    print(header)
    kl_weight = config.kl_weight_target * (min(1.0, epoch / config.kl_anneal_epochs) if config.kl_anneal_epochs > 0 else 1.0)
    for batch_idx, batch in enumerate(train_iter):
        # Initialize torchfold for *encoding*
        enc_fold = FoldExt(cuda=config.cuda)
        enc_fold_nodes = []     # list of fold nodes for encoding
        # Collect computation nodes recursively from encoding process
        for example in batch:
            enc_fold_nodes.append(grassmodel.encode_structure_fold(enc_fold, example))
        # Apply the computations on the encoder model
        enc_fold_nodes = enc_fold.apply(encoder, [enc_fold_nodes])
        # Split into a list of fold nodes per example
        enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)
        # Initialize torchfold for *decoding*
        dec_fold = FoldExt(cuda=config.cuda)
        # Collect computation nodes recursively from decoding process
        box_nodes = []
        sym_nodes = []
        cat_nodes = []
        kld_fold_nodes = []
        for example, fnode in zip(batch, enc_fold_nodes):
            root_code, kl_div = torch.chunk(fnode, 2, 1)
            b_nodes, s_nodes, c_nodes = grassmodel.decode_structure_fold(dec_fold, root_code, example)
            box_nodes.extend(b_nodes)
            sym_nodes.extend(s_nodes)
            cat_nodes.extend(c_nodes)
            kld_fold_nodes.append(kl_div)
            
        apply_lists = []
        if box_nodes: apply_lists.append(box_nodes)
        if sym_nodes: apply_lists.append(sym_nodes)
        if cat_nodes: apply_lists.append(cat_nodes)
        if kld_fold_nodes: apply_lists.append(kld_fold_nodes)
        
        apply_res = dec_fold.apply(decoder, apply_lists)
        
        # Unpack results safely
        res_idx = 0
        device = torch.cuda.current_device() if config.cuda else 'cpu'
        zero_tensor = torch.tensor(0.0, device=device)
        
        if box_nodes:
            box_loss_raw = apply_res[res_idx].sum(dim=0) / len(batch)
            geom_loss = box_loss_raw[0]
            cls_loss = box_loss_raw[1]
            res_idx += 1
        else:
            geom_loss, cls_loss = zero_tensor, zero_tensor
            
        if sym_nodes:
            sym_loss = apply_res[res_idx].sum() / len(batch)
            res_idx += 1
        else:
            sym_loss = zero_tensor
            
        if cat_nodes:
            cat_loss = apply_res[res_idx].sum() / len(batch)
            res_idx += 1
        else:
            cat_loss = zero_tensor
            
        if kld_fold_nodes:
            kldiv_total = apply_res[res_idx].sum().mul(-0.5)
            avg_raw_kld = kldiv_total.item() / len(batch)
            kldiv_loss = kldiv_total.mul(kl_weight) / len(batch)
        else:
            avg_raw_kld = 0.0
            kldiv_loss = zero_tensor
            
        total_loss = (config.lambda_geom * geom_loss + 
                      config.lambda_cls * cls_loss + 
                      config.lambda_sym * sym_loss + 
                      config.lambda_cat * cat_loss + 
                      kldiv_loss)

        # Do parameter optimization
        encoder_opt.zero_grad() 
        decoder_opt.zero_grad()
        total_loss.backward()
        encoder_opt.step() 
        decoder_opt.step()
        # Report statistics
        if batch_idx % config.show_log_every == 0:
            print(log_template.format(strftime("%H:%M:%S", time.gmtime(time.time()-start)),
                epoch+1, config.epochs, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx+len(train_iter)*epoch) / (len(train_iter)*config.epochs),
                geom_loss.item(), cls_loss.item(), sym_loss.item(), cat_loss.item(), avg_raw_kld, total_loss.item()))
        # Plot losses
        if not config.no_plot:
            plot_total_loss[iter_id] = total_loss.item()
            plot_geom_loss[iter_id] = geom_loss.item()
            plot_cls_loss[iter_id] = cls_loss.item()
            plot_sym_loss[iter_id] = sym_loss.item()
            plot_cat_loss[iter_id] = cat_loss.item()
            plot_kldiv_loss[iter_id] = avg_raw_kld
            max_loss = max(max_loss, total_loss.item(), geom_loss.item(), cls_loss.item(), sym_loss.item(), cat_loss.item(), avg_raw_kld)
            dyn_plot.setxlim(0., (iter_id+1)*1.05)
            dyn_plot.setylim(0., max_loss*1.05)
            dyn_plot.update_plots(ydata={'Total_loss':plot_total_loss, 'Geom_loss':plot_geom_loss, 'Cls_loss':plot_cls_loss, 'Sym_loss':plot_sym_loss, 'Cat_loss':plot_cat_loss, 'KLD_loss':plot_kldiv_loss})
            iter_id += 1

    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0 :
        print("Saving snapshots of the models ...... ", end='', flush=True)
        torch.save(encoder, snapshot_folder+'//vae_encoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.item()))
        torch.save(decoder, snapshot_folder+'//vae_decoder_model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.item()))
        print("DONE")
    # Save training log
    if config.save_log and (epoch+1) % config.save_log_every == 0 :
        fd_log = open('training_log.log', mode='a')
        fd_log.write('\nepoch:{} geom:{:.4f} cls:{:.4f} sym:{:.4f} cat:{:.4f} raw_kld:{:.4f} total:{:.4f}'.format(epoch+1, geom_loss.item(), cls_loss.item(), sym_loss.item(), cat_loss.item(), avg_raw_kld, total_loss.item()))
        fd_log.flush()
        fd_log.close()

# Save the final models
print("Saving final models ...... ", end='', flush=True)
torch.save(encoder, config.save_path+'//vae_encoder_model.pkl')
torch.save(decoder, config.save_path+'//vae_decoder_model.pkl')
print("DONE")

if not config.no_plot:
    dyn_plot.save(os.path.join(config.save_path, 'vae_loss_curve.png'))