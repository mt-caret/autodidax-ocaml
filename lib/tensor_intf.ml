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
  val reshape : t -> dims:int array -> t
  val of_float : float -> t
  val arange : int -> t
  val map : t -> f:(float -> float) -> t
  val map2 : t -> t -> f:(float -> float -> float) -> t

  module O : Operators with type t := t
  include Operators with type t := t

  val sin : t -> t
  val cos : t -> t
end
