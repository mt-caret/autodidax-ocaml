open! Core

module Make (S : sig
    val name : string
  end) =
struct
  module T = struct
    type t = int [@@deriving compare]

    let t_of_sexp sexp = snd ([%of_sexp: string * int] sexp)
    let sexp_of_t t = [%sexp_of: string * int] (S.name, t)
    let to_int = Fn.id
  end

  include T
  include Comparable.Make (T)

  let create =
    let counter = ref 0 in
    fun () ->
      let id = !counter in
      incr counter;
      id
  ;;
end
