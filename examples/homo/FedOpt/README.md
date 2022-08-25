# FedOpt example

In the example, we will introduce how to train imagenet with Ifleaner FedOpt and PyTorch.

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

### Result

We train the example [imagenet](../imagenet/README.md) by FedAdam with (learning rate = 1, adaptivity=0.1) and FedAvgM with (learning rate = 1, momentum = 0.9). The results is shown in the table.

<table >
    <tr>
     <th>Fedopt type</th>
      <th>epoch</th>
     <th>client name</th>  
     <th>Top 1 acc</th>
     <th>Top 5 acc</th>  
 </tr >
    <tr >
     <td rowspan="3">FedAdam</td>
          <td rowspan="3">60</td>
     <td>iid-client01</td>
        <td>55.880</td>
        <td>79.550</td>
 </tr>
    <tr>
     <td>iid-client02</td>
        <td>55.920</td>
        <td>79.540</td>
 </tr>
 <tr>
     <td>iid-client03</td>
        <td> 55.730</td>
        <td>79.410</td>
 </tr>

<tr >
     <td rowspan="3">FedAvgM</td>
          <td rowspan="3">60</td>
     <td>iid-client01</td>
        <td>55.960</td>
        <td>79.600</td>
 </tr>
    <tr>
     <td>iid-client02</td>
        <td>55.920</td>
        <td>79.660</td>
 </tr>
 <tr>
     <td>iid-client03</td>
        <td> 56.110</td>
        <td>80.060</td>
 </tr>


<table>
