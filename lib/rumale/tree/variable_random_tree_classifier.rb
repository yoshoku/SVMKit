# frozen_string_literal: true

require 'rumale/tree/decision_tree_classifier'

module Rumale
  module Tree
    # VariableRandomTreeClassifier is a class that implements variable-random tree for classification.
    #
    # @example
    #   estimator =
    #     Rumale::Tree::VariableRandomTreeClassifier.new(
    #       criterion: 'gini', max_depth: 3, max_leaf_nodes: 10, min_samples_leaf: 5,
    #       alpha: 0.3, random_seed: 1
    #     )
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - F. T. Liu, K. M. Ting, Y. Yu, and Z-H. Zhou, "Spectrum of Variable-Random Trees," Journal of Artificial Intelligence Research, vol. 32, pp. 355--384, 2008.
    class VariableRandomTreeClassifier < DecisionTreeClassifier
      # Return the class labels.
      # @return [Numo::Int32] (size: n_classes)
      attr_reader :classes

      # Return the importance for each feature.
      # @return [Numo::DFloat] (size: n_features)
      attr_reader :feature_importances

      # Return the learned tree.
      # @return [Node]
      attr_reader :tree

      # Return the random generator for random selection of feature index.
      # @return [Random]
      attr_reader :rng

      # Return the labels assigned each leaf.
      # @return [Numo::Int32] (size: n_leafs)
      attr_reader :leaf_labels

      # Create a new classifier with variable-random tree.
      #
      # @param criterion [String] The function to evaluate spliting point. Supported criteria are 'gini' and 'entropy'.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, variable-random tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on variable-random tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param alpha [Float] The probability of choosing deterministic test-selection.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: 'gini', max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil,
                     alpha: 0.3, random_seed: nil)
        check_params_numeric_or_nil(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                                    max_features: max_features, random_seed: random_seed)
        check_params_numeric(min_samples_leaf: min_samples_leaf, alpha: alpha)
        check_params_string(criterion: criterion)
        check_params_positive(max_depth: max_depth, max_leaf_nodes: max_leaf_nodes,
                              min_samples_leaf: min_samples_leaf, max_features: max_features)
        keywd_args = method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h
        keywd_args.delete(:alpha)
        super(keywd_args)
        @params[:alpha] = alpha
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VariableRandomTreeClassifier] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        super
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)
        super
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)
        super
      end

      private

      def find_best_split(x, y, impurity)
        if @sub_rng.rand <= @params[:alpha]
          rand_ids.map { |n| [n, *best_split(x[true, n], y, impurity)] }.max_by(&:last)
        else
          n = rand_ids.first
          [n, *best_split_rand(x[true, n], y, impurity)]
        end
      end

      def best_split_rand(features, y, whole_impurity)
        low, hi = Array.new(features.shape[0]) { |n| n }.sample(2, random: @sub_rng)
        threshold = 0.5 * (features[low] + features[hi])
        l_ids = features.le(threshold).where
        r_ids = features.gt(threshold).where
        l_impurity = l_ids.empty? ? 0.0 : impurity(y[l_ids, true])
        r_impurity = r_ids.empty? ? 0.0 : impurity(y[r_ids, true])
        gain = whole_impurity -
               l_impurity * l_ids.size.fdiv(y.shape[0]) -
               r_impurity * r_ids.size.fdiv(y.shape[0])
        [l_impurity, r_impurity, threshold, gain]
      end
    end
  end
end
