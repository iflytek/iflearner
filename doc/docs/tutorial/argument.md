## Startup Options

### Client
You need import `parser` from `iflearner.business.homo.argument` firstly and then call `parser.parse_args`, because we provided some common arguments that
need to be parsered. If you want to add addtional arguments for yourself, you can call `parser.add_argument` repeatedly to add them before `parser.parse_args`
has been called. After parsered arguments, you can create your trainer instance base on previous implemented class, and put it with `args` to `train_client.Controller`. 
In the end, you just need call `controller.run` to run your client.

The list of client default options is as follows:

| option | type |  describe | default |
| :-----| :---- | :---- |:----: |
| name | str | name of client  | client |
| epochs | int | number of total epochs to run | 10 |
| server | str | the address of connecting aggerating server | localhost:50001 |
| enable-ll | int | enable local training (1 、 0) | 0 |
| peers | str | enabled SMPC if the argument had specified. all clients' addresses and use semicolon separate all addresses. First one is your own address.  |  |
| cert | str | path of server SSL cert. use secure channel to connect to server if not none|  |

### Server

The list of server options is as follows:

| option | type |  describe | default |
| :-----| :---- | :---- |:----: |
| num | int | the number of all clients  | 0 |
| epochs | int | the total epoch |  |
| addr | str |The aggregation server itself listens to the address (used for client connections) | "0.0.0.0:50001" |
| http_addr | str |Federation training status listening address (for viewing federation training status) | "0.0.0.0:50002" |
| strategy | str |the aggregation starategy (FedAvg、Scaffold、FedOpt、qFedAvg、FedNova) | FedAvg |
| strategy_params | dict | specify the params of strategy | {} |