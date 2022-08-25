# FedDyn example

In the example, we will introduce how to train imagenet with Ifleaner [FedDyn](https://openreview.net/pdf?id=B7v4QMR6Z9w) and PyTorch.
We customize an aggregation strategy [`feddyn_server.py`](./feddyn_server.py),
### Server

```cmd
python server.py -n 2
```

### Client

```cmd
python imagenet.py --data your_data_path1 --name iid-client01 --epochs 60 --server "0.0.0.0:50001"
python imagenet.py --data your_data_path2 --name iid-client02 --epochs 60 --server "0.0.0.0:50001"
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
     <td rowspan="6">FedAdam</td>
          <td rowspan="6">60</td>
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
<td>noniid-client01</td>
        <td>50.570</td>
        <td>76.380</td>
 </tr>
    <tr>
     <td>noniid-client02</td>
        <td>51.390</td>
        <td>76.880</td>
 </tr>
 <tr>
     <td>noniid-client03</td>
        <td> 51.690</td>
        <td>76.720</td>
 </tr>
<tr >
     <td rowspan="6">FedAvgM</td>
          <td rowspan="6">60</td>
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
