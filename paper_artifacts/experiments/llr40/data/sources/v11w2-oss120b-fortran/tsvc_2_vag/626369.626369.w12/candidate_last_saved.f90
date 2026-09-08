module tsvc_2_vag_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
    implicit none
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    interface
      subroutine tsvc_2_vag_fp64_ref(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64_ref")
        import :: c_double, c_int32_t, c_int64_t
        real(c_double), intent(out) :: a(*)
        real(c_double), intent(in) :: b(*)
        integer(c_int32_t), intent(in) :: ip(*)
        integer(c_int64_t), value :: LEN_1D
      end subroutine tsvc_2_vag_fp64_ref
    end interface
    call tsvc_2_vag_fp64_ref(a, b, ip, LEN_1D)
  end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
