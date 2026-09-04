module argmax_with_index_mod
  use iso_c_binding
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, LEN_1D) bind(C, name="argmax_with_index_fp64")
    ! Compute maximum and zero-based index of an array of double precision values.
    real(c_double), intent(in) :: a(*)                ! Input array (LEN_1D elements)
    integer(c_int64_t), intent(out) :: out_index(1)      ! Output index (zero-based?)
    real(c_double), intent(out) :: out_value          ! Output maximum value
    integer(c_int64_t), value :: LEN_1D               ! Length of the input array (passed by value)
    real(c_double) :: x
    integer(c_int64_t) :: idx
    integer(c_int64_t) :: i
    ! integer(c_int64_t) :: idx_out  ! Not needed
    
    if (LEN_1D <= 0_c_int64_t) then
        out_value = 0.0_c_double
        out_index = 0_c_int64_t
        return
    end if
    
    x = a(1)               ! Corresponds to a[0] in C
    idx = 0_c_int64_t
    
    do i = 1_c_int64_t, LEN_1D - 1_c_int64_t
        if (a(i+1) > x) then
            x = a(i+1)
            idx = i
        end if
    end do
    
    out_value = x
    out_index(1) = idx
  end subroutine argmax_with_index_fp64
end module argmax_with_index_mod
