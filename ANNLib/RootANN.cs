using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace ANNLib
{
    public abstract class RootANN
    {
        public enum ActivationType
        {
            PositiveSygmoid,// Положительная униполярная сигнмоида.
            BipolarSygmoid  // Биполярная сигмоида.
        };

        /**
        * Прочитать нейронную сеть из файла. Сеть сохраняется вызовом метода Save.
        * @param filepath - имя и путь до файла с сетью.
        * @return - успешность считывания.
        */
        public abstract bool Load(string filepath);

        /**
        * Сохранить нейронную сеть в файл. Сеть загружается вызовом метода Load.
        * @param filepath - имя и путь до файла с сетью.
        * @return - успешность сохранения.
        */
        public abstract bool Save(string filepath);


        /**
        * Получить конфигурацию сети.
        * @return конфигурация сети - массив - в каждом элементе хранится количество нейронов в слое.
        *			Номер элемента соответствует номеру слоя.
        */
       // public abstract List<uint> GetConfiguration();

        /**
        * Проинициализирвать веса сети случайным образом.
        */
        public abstract void RandomInit();

        /**************************************************************************/
        /**********************ЭТО ВАМ НАДО РЕАЛИЗОВАТЬ САМИМ**********************/
        /**************************************************************************/

        /**
        * Получить строку с типом сети.
        * @return описание сети, содержит запись о типе нейронной сети и авторе библиотеки.
        */
        public abstract string GetType();

        /**
        * Спрогнозировать выход по заданному входу.
        * @param input - вход, длина должна соответствовать количеству нейронов во входном слое.
        * @return выход сети, длина соответствует количеству нейронов в выходном слое.
        */
        public abstract List<double> Predict(List<double> input);

        /**
        * Создать нейронную сеть
        * @param configuration - конфигурация нейронной сети.
        *   Каждый элемент представляет собой количество нейронов в очередном слое.
        * @param activation_type - тип активационной функции (униполярная, биполярная).
        * @param scale - масштаб активационной функции.
        */
        public abstract ANN.ANeuralNetwork CreateNeuralNetwork(List<uint> configuration, ActivationType activationType,
            double scale);

        /**
        * Обучить сеть методом обратного распространения ошибки.
        * В ходе работы метода, после выполнения обучения флаг is_trained должен устанавливаться в true.
        * @param ann - нейронная сеть, которую необходимо обучить.
        * @param inputs - входы для обучения.
        * @param outputs - выходы для обучения.
        * @param max_iters - максимальное количество итераций при обучении.
        * @param eps - средняя ошибка по всем примерам при которой происходит остановка обучения.
        * @param speed - скорость обучения.
        * @param std_dump - сбрасывать ли информацию о процессе обучения в стандартный поток вывода?
        */
        public abstract double BackPropTraining(List<List<double>> inputs, List<List<Double>> outputs, int maxIters = 10000,
                                          double eps = 0.1, double speed = 0.1, bool std_dump = false);

        /**
        * Провести одну итерацию обучения методом обратного распространения ошибки.
        * @param ann - нейронная сеть, которую необходимо обучить.
        * @param input - вход для обучения.
        * @param outputs - выход для обучения.
        * @param speed - скорость обучения.
        */
        public abstract double BackPropTrainingIteration(List<double> input, List<double> output, double speed);

        /***************************************************************************/
        /***************************************************************************/


        /** 
         * Веса сети. 
         * Первый индекс - номер слоя от которого идёт связь, 
         * второй индекс - номер нейрона от которого идёт связь, 
         * третий индекс - номер нейрона к которому идёт связь. 
         */
        private List<List<List<double>>> weights;

        public List<List<List<double>>> Weights
        {
            get => weights ;
            set => weights = value;
        }

        /**
        * Конфигурация сети.
        * номер элемета в массиве соответсвует номеру слоя.
        * значение - количеству нейронов.
        */
        List<uint> configuration;

        public List<uint> Configuration
        {
            get => configuration;
            set => configuration = value;
        }

        /** Обучена ли сеть? */
        bool isTrained;

        public bool IsTrained
        {
            get => isTrained;
            set => isTrained = value;
        }

        /** Масштабирующий коэффициент аргумента сигмоиды. */
        double scale;

        public double Scale
        {
            get => scale;
            set => scale = value;
        }


        /** Тип активационной функции. */
        ActivationType functionType;

        public ActivationType FunctionType
        {
            get => functionType;
            set => functionType = value;
        }

        /**
        * Вычислить значение активационной функции.
        * @param neuron_input - входное значение нейрона.
        * @return - значение активационной фунции.
        */
        public abstract double Activation(double inputNeuron);

        /**
        * Вычислить значение производной активационной функции.
        * @param activation - значение активационной фнункции, для которой хотим вычислить производную.
        * @return - значение производной активационной фунции.
        */
        public abstract double ActivationDerivative(double activation);


        /**
        * Тестовая функция для проверки подключения библиотеки.
        * @return строка с поздравлениями.
        */

        public abstract string GetTestString();



    }
}
