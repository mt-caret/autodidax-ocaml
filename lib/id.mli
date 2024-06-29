open! Core

module Make (_ : sig
    val name : string
  end) : sig
  type t [@@deriving sexp]

  include Comparable.S with type t := t
  include Hashable.S with type t := t

  val create : unit -> t
  val to_int : t -> int
end
