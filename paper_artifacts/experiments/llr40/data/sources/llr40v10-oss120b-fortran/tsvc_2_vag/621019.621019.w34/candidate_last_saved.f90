module tsvc_2_vag_mod
  use iso_c_binding, only: c_double, c_int32_t, c_int64_t
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*)
    integer(c_int32_t), intent(in) :: ip(*)
    integer(c_int64_t), value :: LEN_1D
    interface
      subroutine tsvc_2_vag_fp64_c(a, b, ip, LEN_1D) bind(C, name="tsvc_2_vag_fp64_c")
        import :: c_double, c_int32_t, c_int64_t
        real(c_double), intent(out) :: a(*)
        real(c_double), intent(in) :: b(*)
        integer(c_int32_t), intent(in) :: ip(*)
        integer(c_int64_t), value :: LEN_1D
      end subroutine tsvc_2_vag_fp64_c
    end interface
    call tsvc_2_vag_fp64_c(a, b, ip, LEN_1D)
  end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
