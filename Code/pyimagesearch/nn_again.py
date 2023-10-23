import numpy as np

class NeuralNetwork:
    '''
        - Bước 1: tạo một constructor với 2 tham số được truyền vào là 'layers' (đại diện cho cấu trúc mạng), và 'alpha' (đại diện cho 
        learinng rate) với giá trị mặc định là 0.1. 
        - Bước 2: khởi tạo các thuộc tính cho lớp Neural Network bao gồm: 1 list 'W' lưu các trọng số, 1 list 'layers' lưu:  kiến trúc mạng, 
        1 số thực 'alpha' để lưu tốc độ học
        - Bước 3: duyệt qua từ lớp đầu tiên đến trước 2 lớp cuối
            + 3.1: dùng hàm random.rand để tạo ra một ma trận 'w' với kích thước m*n với m là số lượng node của lớp hiện tại (bao gồm cả bias),
            n là số lượng node của lớp tiếp theo (bao gồm cả bias)
            + 3.2: sau khi tạo ra ma trận 'w', scale nó bằng cách chia cho căn bậc 2 của số node của lớp hiện tại
            + 3.2: thực hiện thêm vào list 'W'
        - Bước 4: khởi tạo 1 ma trận tham số với kích thước a*b với a là số lượng node của lớp trước lớp cuối (bao gồm cả bias), 
        b là số lượng node của output layer(lớp cuối) - ngoại trừ bias
        
    '''
    def __init__(self, alpha = 0.1, layers):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):
            w = np.random.rand(layers[i] + 1, layers[i+1] + 1)
            w  = w / np.sqrt(layers[i])
            self.W.append(w)

        w = np.random.rand(layers[i-2] + 1, layers[-1])
        self.W.append(w)

    '''
    Viết magic method repr để hiển thị cấu trúc mạng:
        + Bước 1: khởi tạo phương thức repr
        + Bước 2: trả về một chuối được format lại
    '''
    def __repr__(self):
        return "Architecture of NN: {}".format('-'.join(str(l) for l in self.layers))

    '''
    Viết activation function sigmoid:
        + Bước 1: khởi tạo hàm 'sigmoid' với 1 tham số 'x' là tổng hợp output của lớp trước đó cho node sau với công thức: output = wx + b
        + Bước 2: trả về giá trị đã qua ánh xạ bằng sigmoid function với công thức 1/(1 + e^-x)
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    '''
    Viết hàm 'calculate_loss' để thực hiện kiểm tra lỗi:
        + Bước 1: Khởi tạo hàm 'calculate_loss' với 2 tham số: 'X' là input, 'targets' là label
        + Bước 2: chuyển mảng 'targets' thành mảng 2d bằng hàm np.atleast2d
        + Bước 3: thực hiện predict 'X' input
        + Bước 4: tính loss bằng công thức 1/2 * sum ((targets - predict)^2)
        + Bước 5: trả về loss
    '''

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(X)
        predict = self.predict(X, addBias=False)
        loss = 1/2 * (np.sum((targets - predict) ** 2))
        return loss

    '''
    Viết hàm 'predict' để thực hiện dự đoán:
        - Bước 1: khai báo hàm 'predict' với 2 tham số: 'X' là input, và tham số 'addBias' (với giá trị mặc định là True)
        - Bước 2: sử dụng hàm np.atleast2d để chuyển input thành mảng ít nhất là 2 chiều
        - Bước 3: Nếu addBias bằng True, nó sẽ thực hiện thêm vào mảng toàn số 1 theo chiều ngang vào x
        - Bước 4: duyệt qua tất cả layer 
            + 4.1: thực hiện feed forward cho x: theo công thức output = sigmoid(wx + b)
        - Bước 5: trả về giá trị dự đoán 
    '''

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        for layer in np.arange(0, len(layer)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
    
        return p


    '''
    Thực hiện training model bằng hàm fit():
        - Bước 1: khai báo hàm 'fit' với các 4 tham số đầu vào: X là tập feature, y là tập label, 'epochs' là số lần huấn luyện 
        với giá trị mặc định là 1000, 'displayUpdate' với giá trị mặc định là 100, tức là cứ sau 100 epochs sẽ hiện một lần loss
        - Bước viết thiếu: concatenate mảng 'X' với mảng 0 
        - Bước 2: duyệt qua từng epoch
            + Bước 2.1: duyệt song song từng cặp feature, label:
                + Bước 2.1.1: gọi hàm fit_partial với 2 tham số là 'X' và 'target'
            + Bước 2.2: kiểm tra điều kiện nếu epoch = 0 hoặc epoch thứ i + 1 chia hết cho displayUpdate thì:
                + Bước 2.2.1: gọi hàm 'calculate_loss' để tính loss
                + Bước 2.2.2: in ra thông tin về số thứ tự epoch, loss của epoch đó
    '''

    def fit(self, X, y, epochs = 1000, displayUpdate = 100):
        X = np.c_(X, np.ones((X.shape[0])))
        for epoch in np.arange(0, len(epochs)):
            for x,target in zip(X,y):
                self.fit_partial(x,target)
            
            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print('[INFO] - epoch:{}-loss:{}'.format(epoch+1, loss))


    '''
    Viết hàm 'fit_feature' để thực hiện nhiệm vụ backpropagation
        Bước 1: khai báo hàm 'fit_partial' với 2 tham số 'X' và 'y'
        Bước 2: khởi tạo list A là danh sách các output của các activation function ở mỗi layer, với giá trị 
        đầu tiên là input feature vector
        Bước 3: duyệt qua từng weight trong danh sách 'W':
            + Bước 3.1: tính 'net'
            + Bước 3.2: tạo biến 'out' có được bằng ánh xạ sigmoid function cho 'net'
            + Bước 3.3: thêm 'out' vào list 'A'
            *** Bắt đầu feed forward:
            + Bước 3.4: tính 'error' bằng cách lấy phần tử cuối cùng của list 'A' - 'y'
            + Bước 3.5: 
    '''

        
    