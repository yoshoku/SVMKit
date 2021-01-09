# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/decomposition/pca'
require 'rumale/pairwise_metric'
require 'rumale/utils'
require 'lbfgsb'

module Rumale
  module MetricLearning
    # MLKR is a class that implements Metric Learning for Kernel Regression.
    #
    # @example
    #   transformer = Rumale::MetricLearning::MLKR.new
    #   transformer.fit(training_samples, traininig_target_values)
    #   low_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Weinberger, K. Q. and Tesauro, G., "Metric Learning for Kernel Regression," Proc. AISTATS'07, pp. 612--629, 2007.
    class MLKR
      include Base::BaseEstimator
      include Base::Transformer

      # Returns the metric components.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with MLKR.
      #
      # @param n_components [Integer] The number of components.
      # @param init [String] The initialization method for components ('random' or 'pca').
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      #   This value is given as tol / Lbfgsb::DBL_EPSILON to the factr argument of Lbfgsb.minimize method.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      #   If true is given, 'iterate.dat' file is generated by lbfgsb.rb.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: nil, init: 'random', max_iter: 100, tol: 1e-6, verbose: false, random_seed: nil)
        check_params_numeric_or_nil(n_components: n_components, random_seed: random_seed)
        check_params_numeric(max_iter: max_iter, tol: tol)
        check_params_string(init: init)
        check_params_boolean(verbose: verbose)
        @params = {}
        @params[:n_components] = n_components
        @params[:init] = init
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @components = nil
        @n_iter = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [MLKR] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        n_features = x.shape[1]
        n_components = if @params[:n_components].nil?
                         n_features
                       else
                         [n_features, @params[:n_components]].min
                       end
        @components, @n_iter = optimize_components(x, y, n_features, n_components)
        @prototypes = x.dot(@components.transpose)
        @values = y
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples]) The target values to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_tvalue_array(y)
        check_sample_tvalue_size(x, y)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = check_convert_sample_array(x)
        x.dot(@components.transpose)
      end

      private

      def init_components(x, n_features, n_components)
        if @params[:init] == 'pca'
          pca = Rumale::Decomposition::PCA.new(n_components: n_components)
          pca.fit(x).components.flatten.dup
        else
          Rumale::Utils.rand_normal([n_features, n_components], @rng.dup).flatten.dup
        end
      end

      def optimize_components(x, y, n_features, n_components)
        # initialize components.
        comp_init = init_components(x, n_features, n_components)
        # initialize optimization results.
        res = {}
        res[:x] = comp_init
        res[:n_iter] = 0
        # perform optimization.
        verbose = @params[:verbose] ? 1 : -1
        res = Lbfgsb.minimize(
          fnc: method(:mlkr_fnc), jcb: true, x_init: comp_init, args: [x, y],
          maxiter: @params[:max_iter], factr: @params[:tol] / Lbfgsb::DBL_EPSILON, verbose: verbose
        )
        # return the results.
        n_iter = res[:n_iter]
        comps = n_components == 1 ? res[:x].dup : res[:x].reshape(n_components, n_features)
        [comps, n_iter]
      end

      def mlkr_fnc(w, x, y)
        # initialize some variables.
        n_features = x.shape[1]
        n_components = w.size / n_features
        # projection.
        w = w.reshape(n_components, n_features)
        z = x.dot(w.transpose)
        # predict values.
        kernel_mat = Numo::NMath.exp(-Rumale::PairwiseMetric.squared_error(z))
        kernel_mat[kernel_mat.diag_indices] = 0.0
        norm = kernel_mat.sum(1)
        norm[norm.eq(0)] = 1
        y_pred = kernel_mat.dot(y) / norm
        # calculate loss.
        y_diff = y_pred - y
        loss = (y_diff**2).sum
        # calculate gradient.
        weight_mat = y_diff * y_diff.expand_dims(1) * kernel_mat
        weight_mat = weight_mat.sum(0).diag - weight_mat
        gradient = 8 * z.transpose.dot(weight_mat).dot(x)
        [loss, gradient.flatten.dup]
      end
    end
  end
end