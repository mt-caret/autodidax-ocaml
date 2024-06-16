open! Core

module type Operators = sig
  type t

  val ( + ) : t -> t -> t
  val ( - ) : t -> t -> t
  val ( * ) : t -> t -> t
  val ( ~- ) : t -> t
end

module type Tensor = sig
  type t [@@deriving sexp_of]

  val dims : t -> int array
  val length : t -> int
  val item : t -> float
  val get : t -> int array -> float
  val set : t -> int array -> float -> unit
  val fill : t -> float -> unit
  val reshape : t -> dims:int array -> t
  val of_float : float -> t
  val create : dims:int array -> float -> t
  val zeros : dims:int array -> t
  val ones : dims:int array -> t
  val arange : int -> t
  val of_xla_literal : Xla.Literal.t -> t
  val map : t -> f:(float -> float) -> t
  val map2 : t -> t -> f:(float -> float -> float) -> t

  module O : Operators with type t := t
  include Operators with type t := t

  val sin : t -> t
  val cos : t -> t
end
