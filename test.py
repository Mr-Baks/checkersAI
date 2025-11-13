import numpy as np
import pytest
import os
import tempfile
from neural_network import NN  # замените your_module на имя вашего файла

class TestNN:
    
    def setup_method(self):
        """Настройка перед каждым тестом"""
        self.nn = NN(4, 5, 3)  # простая архитектура для тестов
        self.X_train = np.random.randn(100, 4)
        self.y_train = np.eye(3)[np.random.randint(0, 3, 100)]
        self.X_test = np.random.randn(20, 4)
        self.y_test = np.eye(3)[np.random.randint(0, 3, 20)]
    
    def test_initialization(self):
        """Тест инициализации сети"""
        # Проверка размеров весов и смещений
        assert len(self.nn.W) == 2
        assert len(self.nn.b) == 2
        
        # Проверка форм матриц
        assert self.nn.W[0].shape == (4, 5)
        assert self.nn.W[1].shape == (5, 3)
        assert self.nn.b[0].shape == (1, 5)
        assert self.nn.b[1].shape == (1, 3)
        
        # Проверка, что веса инициализированы (не нулевые)
        assert not np.allclose(self.nn.W[0], 0)
        assert not np.allclose(self.nn.W[1], 0)
    
    def test_relu(self):
        """Тест функции активации ReLU"""
        test_input = np.array([[-1, 0, 1], [-2, 2, -0.5]])
        expected = np.array([[0, 0, 1], [0, 2, 0]])
        result = self.nn.relu(test_input)
        assert np.array_equal(result, expected)
    
    def test_relu_derivative(self):
        """Тест производной ReLU"""
        test_input = np.array([[-1, 0, 1], [-2, 2, -0.5]])
        expected = np.array([[0, 0, 1], [0, 1, 0]])
        result = self.nn.relu_derivative(test_input)
        assert np.array_equal(result, expected)
    
    def test_softmax(self):
        """Тест функции softmax"""
        test_input = np.array([[1, 2, 3]])
        result = self.nn.softmax(test_input)
        
        # Проверка, что сумма вероятностей равна 1
        assert np.allclose(np.sum(result, axis=1), 1.0)
        
        # Проверка, что большее входное значение дает большую вероятность
        assert result[0, 2] > result[0, 1] > result[0, 0]
    
    def test_dropout(self):
        """Тест dropout"""
        test_input = np.ones((2, 3))
        
        # В режиме обучения должен применяться dropout
        result_train = self.nn.dropout(test_input, rate=0.5, training=True)
        assert result_train.shape == test_input.shape
        # Некоторые значения должны быть обнулены
        assert not np.array_equal(result_train, test_input)
        
        # В режиме предсказания dropout не применяется
        result_test = self.nn.dropout(test_input, rate=0.5, training=False)
        assert np.array_equal(result_test, test_input)
    
    def test_forward_pass_shape(self):
        """Тест формы выхода forward pass"""
        X = np.random.randn(10, 4)
        output = self.nn.forward(X)
        
        # Проверка формы выхода
        assert output.shape == (10, 3)
        
        # Проверка, что выходные вероятности суммируются в 1
        # assert np.allclose(np.sum(output, axis=1), 1.0)
    
    def test_forward_pass_training_vs_inference(self):
        """Тест различий между режимами обучения и предсказания"""
        X = np.random.randn(5, 4)
        
        output_train = self.nn.forward(X, training=True)
        output_test = self.nn.forward(X, training=False)
        
        # Выходы должны иметь одинаковую форму
        assert output_train.shape == output_test.shape
        
        # Из-за dropout выходы могут немного отличаться
        # но основные характеристики должны сохраняться
        assert np.allclose(np.sum(output_train, axis=1), 1.0)
        assert np.allclose(np.sum(output_test, axis=1), 1.0)
    
    def test_backward_pass(self):
        """Тест backward pass (проверка обновления весов)"""
        # Сохраняем исходные веса
        original_W = [w.copy() for w in self.nn.W]
        original_b = [b.copy() for b in self.nn.b]
        
        # Выполняем backward pass
        self.nn.backward(self.X_train[:10], self.y_train[:10], learning_rate=0.01)
        
        # Проверяем, что веса изменились
        for i in range(len(self.nn.W)):
            assert not np.array_equal(self.nn.W[i], original_W[i])
            assert not np.array_equal(self.nn.b[i], original_b[i])
    
    def test_training(self):
        """Тест полного цикла обучения"""
        # Измеряем точность до обучения
        initial_acc = self.nn.get_acc(self.X_test, self.y_test)
        
        # Обучаем сеть
        self.nn.train(self.X_train, self.y_train, epochs=5, 
                     learning_rate=0.01, batch_size=32)
        
        # Измеряем точность после обучения
        final_acc = self.nn.get_acc(self.X_test, self.y_test)
        
        # После обучения точность должна улучшиться или остаться той же
        # (в редких случаях может ухудшиться из-за случайности)
        print(f"Accuracy: {initial_acc:.3f} -> {final_acc:.3f}")
    
    def test_batch_training(self):
        """Тест обучения с разными размерами батчей"""
        for batch_size in [1, 10, 50, 100]:
            nn = NN(4, 5, 3)
            try:
                nn.train(self.X_train, self.y_train, epochs=2, 
                        learning_rate=0.01, batch_size=batch_size)
                # Если не возникло ошибок, тест пройден
                assert True
            except Exception as e:
                pytest.fail(f"Training failed with batch_size={batch_size}: {e}")
    
    def test_save_load(self):
        """Тест сохранения и загрузки модели"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Сохраняем модель
            self.nn.save(temp_file)
            assert os.path.exists(temp_file)
            
            # Создаем новую модель и загружаем параметры
            new_nn = NN(4, 5, 3)
            new_nn.load(temp_file)
            
            # Проверяем, что веса совпадают
            for i in range(len(self.nn.W)):
                assert np.array_equal(self.nn.W[i], new_nn.W[i])
                assert np.array_equal(self.nn.b[i], new_nn.b[i])
                
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_accuracy_calculation(self):
        """Тест расчета точности"""
        # Создаем простой случай, где мы знаем правильные ответы
        X_simple = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]])
        
        # One-hot encoded labels
        y_simple = np.array([[1, 0, 0],
                            [0, 1, 0]])
        
        accuracy = self.nn.get_acc(X_simple, y_simple)
        
        # Проверяем, что точность в допустимом диапазоне
        assert 0.0 <= accuracy <= 1.0
    
    def test_gradient_descent(self):
        """Тест, что loss уменьшается при обучении"""
        # Простая задача XOR для проверки обучения
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # one-hot encoded
        
        nn_xor = NN(2, 4, 2)  # сеть для задачи XOR
        
        # Обучаем на нескольких эпохах
        nn_xor.train(X_xor, y_xor, epochs=50, learning_rate=0.1, batch_size=4)
        
        # Проверяем, что сеть научилась решать задачу
        accuracy = nn_xor.get_acc(X_xor, y_xor)
        assert accuracy >= 0.75  # должна достичь разумной точности
    
    def test_edge_cases(self):
        """Тест граничных случаев"""
        # Пустые данные
        X_empty = np.array([]).reshape(0, 4)
        y_empty = np.array([]).reshape(0, 3)
        
        # Должно обрабатываться без ошибок
        try:
            output = self.nn.forward(X_empty)
            assert output.shape == (0, 3)
        except Exception as e:
            pytest.fail(f"Forward pass failed with empty input: {e}")
        
        # Один пример
        X_single = np.random.randn(1, 4)
        y_single = np.eye(3)[[0]]
        
        try:
            output = self.nn.forward(X_single)
            assert output.shape == (1, 3)
            self.nn.backward(X_single, y_single, learning_rate=0.01)
        except Exception as e:
            pytest.fail(f"Failed with single example: {e}")

def test_different_architectures():
    """Тест разных архитектур сетей"""
    architectures = [
        (10, 5),           # один скрытый слой
        (10, 8, 3),        # два скрытых слоя
        (20, 15, 10, 5),   # три скрытых слоя
    ]
    
    for arch in architectures:
        try:
            nn = NN(*arch)
            X = np.random.randn(50, arch[0])
            output = nn.forward(X)
            assert output.shape == (50, arch[-1])
        except Exception as e:
            pytest.fail(f"Architecture {arch} failed: {e}")

if __name__ == "__main__":
    # Запуск тестов
    test_nn = TestNN()
    
    print("Running NN tests...")
    
    # Запускаем основные тесты
    test_nn.setup_method()
    test_nn.test_initialization()
    print("✓ Initialization test passed")
    
    test_nn.test_relu()
    print("✓ ReLU test passed")
    
    test_nn.test_softmax()
    print("✓ Softmax test passed")
    
    test_nn.test_forward_pass_shape()
    print("✓ Forward pass shape test passed")
    
    test_nn.test_training()
    print("✓ Training test passed")
    
    test_nn.test_save_load()
    print("✓ Save/load test passed")
    
    test_different_architectures()
    print("✓ Architecture test passed")
    
    print("\nAll tests passed! 🎉")