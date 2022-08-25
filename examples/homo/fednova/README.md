# FedNova example

In the example, we will introduce how to train imagenet with Ifleaner FedNova and PyTorch.

### Server

```cmd
python server.py -n 3 
```

### Client

```cmd
python imagenet.py --data your_data_path1 --name iid-client01 --epochs 60 --server "0.0.0.0:50001"
python imagenet.py --data your_data_path2 --name iid-client02 --epochs 60 --server "0.0.0.0:50001"
python imagenet.py --data your_data_path3 --name iid-client03 --epochs 60 --server "0.0.0.0:50001"
```

The client only use vanilla SGD as a local solve, so this situation is about equal to Fedavg. If you want to use a full reproduced method of origin paper, you can modify the Optimizer referring to the author's open source [code](https://github.com/JYWa/FedNova.git) by yourself.

### Result

We train the example [imagenet](../imagenet/README.md) by FedNova. The results is shown in the table.

<table >
    <tr>
     <th>Fedopt type</th>
      <th>epoch</th>
     <th>client name</th>  
     <th>Top 1 acc</th>
     <th>Top 5 acc</th>  
 </tr >
    <tr >
     <td rowspan="6">FedNova</td>
          <td rowspan="6">60</td>
     <td>noniid-client01</td>
        <td>46.130</td>
        <td>73.580</td>
 </tr>
    <tr>
     <td>noniid-client02</td>
        <td>46.700</td>
        <td>73.480</td>
 </tr>
 <tr>
     <td>noniid-client03</td>
        <td>46.600</td>
        <td>73.640</td>
 </tr>


<table>
