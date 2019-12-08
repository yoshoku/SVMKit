# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/tree/node'

module Rumale
  # This module consists of the classes that implement tree models.
  module Tree
    # BaseDecisionTree is an abstract class for implementation of decision tree-based estimator.
    # This class is used internally.
    class BaseDecisionTree
      include Base::BaseEstimator

      # Initialize a decision tree-based estimator.
      #
      # @param criterion [String] The function to evalue spliting point.
      # @param max_depth [Integer] The maximum depth of the tree.
      #   If nil is given, decision tree grows without concern for depth.
      # @param max_leaf_nodes [Integer] The maximum number of leaves on decision tree.
      #   If nil is given, number of leaves is not limited.
      # @param min_samples_leaf [Integer] The minimum number of samples at a leaf node.
      # @param max_features [Integer] The number of features to consider when searching optimal split point.
      #   If nil is given, split process considers all features.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      #   It is used to randomly determine the order of features when deciding spliting point.
      def initialize(criterion: nil, max_depth: nil, max_leaf_nodes: nil, min_samples_leaf: 1, max_features: nil, random_seed: nil)
        @params = {}
        @params[:criterion] = criterion
        @params[:max_depth] = max_depth
        @params[:max_leaf_nodes] = max_leaf_nodes
        @params[:min_samples_leaf] = min_samples_leaf
        @params[:max_features] = max_features
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @tree = nil
        @feature_importances = nil
        @n_leaves = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Return the index of the leaf that each sample reached.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Leaf index for sample.
      def apply(x)
        x = check_convert_sample_array(x)
        Numo::Int32[*(Array.new(x.shape[0]) { |n| apply_at_node(@tree, x[n, true]) })]
      end

      private

      def apply_at_node(node, sample)
        return node.leaf_id if node.leaf
        return apply_at_node(node.left, sample) if node.right.nil?
        return apply_at_node(node.right, sample) if node.left.nil?
        if sample[node.feature_id] <= node.threshold
          apply_at_node(node.left, sample)
        else
          apply_at_node(node.right, sample)
        end
      end

      def build_tree(x, y)
        y = y.expand_dims(1).dup if y.shape[1].nil?
        @feature_ids = Array.new(x.shape[1]) { |v| v }
        @tree = grow_node(0, x, y, impurity(y))
        @feature_ids = nil
        nil
      end

      def grow_node(depth, x, y, impurity)
        # intialize node.
        n_samples = x.shape[0]
        node = Node.new(depth: depth, impurity: impurity, n_samples: n_samples)

        # terminate growing.
        unless @params[:max_leaf_nodes].nil?
          return nil if @n_leaves >= @params[:max_leaf_nodes]
        end

        return nil if n_samples < @params[:min_samples_leaf]
        return put_leaf(node, y) if n_samples == @params[:min_samples_leaf]

        unless @params[:max_depth].nil?
          return put_leaf(node, y) if depth == @params[:max_depth]
        end

        return put_leaf(node, y) if stop_growing?(y)

        # calculate optimal parameters.
        feature_id, left_imp, right_imp, threshold, gain = find_best_split(x, y, impurity)

        return put_leaf(node, y) if gain.nil? || gain.zero?

        left_ids = x[true, feature_id].le(threshold).where
        right_ids = x[true, feature_id].gt(threshold).where
        node.left = grow_node(depth + 1, x[left_ids, true], y[left_ids, true], left_imp)
        node.right = grow_node(depth + 1, x[right_ids, true], y[right_ids, true], right_imp)

        return put_leaf(node, y) if node.left.nil? && node.right.nil?

        node.feature_id = feature_id
        node.threshold = threshold
        node.leaf = false
        node
      end

      def stop_growing?(_y)
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      def put_leaf(_node, _y)
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      def rand_ids
        @feature_ids.sample(@params[:max_features], random: @sub_rng)
      end

      def find_best_split(x, y, impurity)
        rand_ids.map { |n| [n, *best_split(x[true, n], y, impurity)] }.max_by(&:last)
      end

      def best_split(_features, _y, _impurity)
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      def impurity(_y)
        raise NotImplementedError, "#{__method__} has to be implemented in #{self.class}."
      end

      def eval_importance(n_samples, n_features)
        @feature_importances = Numo::DFloat.zeros(n_features)
        eval_importance_at_node(@tree)
        @feature_importances /= n_samples
        normalizer = @feature_importances.sum
        @feature_importances /= normalizer if normalizer > 0.0
        nil
      end

      def eval_importance_at_node(node)
        return nil if node.leaf
        return nil if node.left.nil? || node.right.nil?
        gain = node.n_samples * node.impurity -
               node.left.n_samples * node.left.impurity -
               node.right.n_samples * node.right.impurity
        @feature_importances[node.feature_id] += gain
        eval_importance_at_node(node.left)
        eval_importance_at_node(node.right)
      end
    end
  end
end
